# LiDAR BEV 特征输入  
# → 特征预处理（共享卷积+展平）  
# → 热图生成与峰值筛选（NMS）  
# → Top-K 候选框筛选（类别+位置索引解析）  
# → 候选框特征准备（语义+类别嵌入+位置）  
# → 多层 Transformer 解码（迭代优化特征）  
# → 预测头生成目标参数（中心/尺寸/旋转角/类别）  
# → 训练：损失计算（热图+分类+回归）→ 反向优化  
# → 推理：参数解码+NMS → 最终 3D 检测结果


import copy

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.core import (
    PseudoSampler,
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)

__all__ = ["TransFusionHead"]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@HEADS.register_module()
class TransFusionHead(nn.Module):
    def __init__( #接收配置文件中传递的一些参数
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        num_heads=8,
        nms_kernel_size=1,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        # config for FFN
        common_heads=dict(), #接收配置文件传递的参数 来进行。
        num_heatmap_convs=2,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d"),
        bias="auto",
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_iou=dict(
            type="VarifocalLoss", use_sigmoid=True, iou_weighted=True, reduction="mean"
        ),
        loss_bbox=dict(type="L1Loss", reduction="mean"),
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(TransFusionHead, self).__init__()
        #存储配置文件中传递的参数
        self.fp16_enabled = False

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type="Conv2d"),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
            )
        )
        layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        self.heatmap_head = nn.Sequential(*layers)  #热图头构建
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        #解码器构建
        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel,
                    num_heads,
                    ffn_channel,
                    dropout,
                    activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                )
            )

        # Prediction Head 预测头构建
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                FFN(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                )
            )

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]  #计算  1440  1440   除  8  刚到等于180  180  特征图的形状尺寸
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)  #输入特征图x和y尺寸 180  180    计算位置坐标 编码

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]  #起始点 重点 网格个数
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid] #从上面逐个取出
        )
        
        #[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]：对 meshgrid 的每个方向生成均匀分布的坐标点。
        # torch.meshgrid(...)：将 x 和 y 方向的一维坐标组合成二维网格，生成两个x_size x y_size的矩阵：
        # batch_x：每行都是 x 方向的坐标（横向重复）；
        # batch_y：每列都是 y 方向的坐标（纵向重复）
        
        batch_x = batch_x + 0.5 #加0.5进行偏移
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]    # 拼接x和y坐标，增加批次维度
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)                # 展平空间维度，调整为[批次, 网格总数, 2]格式
        return coord_base  #展平空间维度，调整为[批次, 网格总数, 2]格式

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, img_inputs, metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W]                                                              #生成序列特征
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  #调用生成网格坐标  并复制batch_size次   

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat) #利用特征 生成密集热图  4 128 180 180 ————4 10(对应种类） 180  180
        dense_heatmap_img = None
        heatmap = dense_heatmap.detach().sigmoid() #对每个通道（每个种类） 每个网格 采用sigmoid进行激活  
        padding = self.nms_kernel_size // 2  #nms核
        local_max = torch.zeros_like(heatmap) #复制一个核热图同shape的全0张量数据
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )#提取内部的最大值  通多最大池化  nms卷积核
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner #将内部最大值  放到全0张量中 得到带有卷积后的张量
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[
                :,
                8,
            ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                9,
            ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            local_max[
                :,
                1,
            ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                2,
            ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max) #进一步相乘  0乘仍然为0  最大值有数字表示
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1) #展平 4 10 32400

        # top #num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
            ..., : self.num_proposals
        ]#展平4 324000  然后对最后一个维度进行降序排列  取出前self.num_proposals个 当作候选框
        
        top_proposals_class = top_proposals // heatmap.shape[-1]  #类别信息 通过提案的索引 除热图总数====类别信息  类别是按照通道算的 每个通道32400个 依次排序
        top_proposals_index = top_proposals % heatmap.shape[-1]  #位置信息  50000%32400=17600 → 类别 1 的第 17600 个位置  但是表达的shape是[4,128]  128个框  都是对应0---324000中的128个内容
     
        
        
        query_feat = lidar_feat_flatten.gather(  #生成查询特征  
            index=top_proposals_index[:, None, :].expand(                                              
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        # top_proposals_index[:, None, :]——增加中间维度后：[2, 128]——[2, 1, 128]
        # expand(-1, lidar_feat_flatten.shape[1], -1)   [2, 1, 128]——[2, 128, 128]                  -1是表示不变的内容



        
        self.query_labels = top_proposals_class                                 #存储语义标签 提取候选框的语义标签[4,128]  128个数字  0-9对应十个种类

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(         #利用的提议类别索引信息  生成独热编码  
            0, 2, 1
        )
        #输入top_proposals_class  [batch_size, num_proposals]如[2, 128]  将这个纯数字 转化为独热编码表示 
        #输出形状：[batch_size, num_proposals, num_classes]（如[2, 10, 128]），每个位置是一个num_classes维的独热向量。  128行  10列 吗？
        #示例
            #top_proposals_class的某一行是 [0, 1, 2]（3 个候选框，分别是车、行人、自行车）
            #F.one_hot  表示为[
                                 [1, 0, 0],  # 类别0（车）的独热向量
                                 [0, 1, 0],  # 类别1（行人）的独热向量
                                 [0, 0, 1]   # 类别2（自行车）的独热向量
                                ]
        #permute(0, 2, 1)：调整维度顺序[2, 10, 128]
        #“行人（类别 1）”，其独热向量是[0,1,0,...,0]




        query_cat_encoding = self.class_encoding(one_hot.float())                  #输入独热编码 —— 生成分类的高维度数据    [2, 10, 128]——[2, 128, 128]
        query_feat += query_cat_encoding           #同维度相加                                #将类别信息核特征信息结合

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]  #同样是[4,128]---[4,1,128]
            .permute(0, 2, 1)                               [4,1,128]--[4,128,1]
            .expand(-1, -1, bev_pos.shape[-1]),    #[4,128,1]---[4,128,2]
            dim=1,
        ) #输出形状为 [batch_size, num_proposals, 2]（如 [4, 128, 2]）
        #gather 就是 “索引上面的index，从花名册中把这 128 个位置从bev_pos中挑选出来”，最终得到的 query_pos 就是这 128 个的坐标列表。 得到[4, 128, 2]  得到具体的坐标信息
        
        将bev_pos和里面的内容 在1维度上进行
        
        
        #从预先生成的 BEV 位置嵌入bev_pos属性为展平空间维度，调整为[批次, 网格总数, 2]格式
        中，精准提取出 Top-K 候选框对应的空间坐标（x,y），最终生成每个候选框的位置嵌入（query_pos）
          
        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        # 通过多层 Transformer 解码器对候选框特征（query_feat）进行迭代优化
        # 结合特征（作为注意力的 Key 和 Value）生成越来越精确的目标预测结果（如中心坐标、尺寸等）
        # 并最终返回预测结果用于损失计算或推理
        # 代码分为三个核心部分：
        # LiDAR 特征作为 K/V 的 Transformer 解码器迭代优化：通过多层解码器，利用 LiDAR 的 BEV 特征优化候选框特征；
        # 预测头（Prediction Head）生成具体预测结果：每层解码器输出后，通过预测头生成边界框、类别等具体预测；
        # 结果处理与返回：根据是否需要辅助监督（auxiliary），返回最后一层结果或所有层结果
        # #################################
        
        ret_dicts = []   #是 “return” 的缩写 存储返回结果的列表  ret_dicts即 “要返回的字典列表”，用于存储每一层解码器输出的预测结果（每个结果是一个字典）
        
        for i in range(self.num_decoder_layers):  #逐个遍历解码器层： 多层迭代优化
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"   

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](                              #第 i 层 Transformer 解码器（包含自注意力和交叉注意力模块）
                query_feat, lidar_feat_flatten, query_pos, bev_pos
            )
            #输入参数：查询特征[4, 128, 128] ：形状[4, 128, 128]（4 个样本，128 维特征，128 个候选框），作为注意力的 “查询（Query）”
                       特征序列[4 128 32400]：展平的 LiDAR BEV 特征，形状[4, 128, 32400]（4 个样本，128 维特征，32400 个 BEV 网格位置），作为注意力的 “键（Key）” 和 “值（Value）”—— 代表 “全局场景信息”
                      #查询位置[4, 128, 2] 形状[4, 128, 2]（4 个样本，128 个候选框，每个候选框的 (x,y) 坐标）—— 告诉模型 “候选框当前在哪里”
                      #bev_pos所有 BEV 网格的位置嵌入，形状[4, 32400, 2]（4 个样本，32400 个网格，每个网格的 (x,y) 坐标）—— 辅助模型理解 “全局空间位置”。
            解码器的核心作用：
                        #自注意力（Self-Attention）：让 128 个候选框之间相互 “关注”（例如：“候选框 A 和候选框 B 距离很近，可能是同一类目标”），学习候选框间的关系
                        #交叉注意力（Cross-Attention）：让每个候选框 “聚焦” 到 LiDAR 特征中与自己相关的区域（例如：“车的候选框重点关注 BEV 中对应位置的点云密集区”），从全局特征中吸收细节信息。
            
            #输出：优化后的query_feat（仍为[4, 128, 128]），但比输入更 “精准”—— 包含了更多全局上下文和候选框关系信息。


            
            # Prediction————从特征到具体目标参数
            res_layer = self.prediction_heads[i](query_feat)   
                                #字典类型，存储该层的预测结果，包含的关键键值对（结合维度）
                                    #预测结果：center  size    heading   cls_score   四个内容 详细看预测头的内容
                                #self.prediction_heads[i]：第 i 层解码器对应的预测头（由卷积层组成的小型网络），
                                #作用是将抽象的query_feat（128 维特征）转换为具体的 3D 目标参数。       
                                #res_layer——字典类型 通过键访问内容
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
                                #计算绝对中心坐标
                                #query_pos.permute(0, 2, 1)：将query_pos的形状从[4, 128, 2]（样本数，候选框数，x/y）调整为[4, 2, 128]（样本数，x/y, 候选框数）
                                #与res_layer["center"]的维度（[4, 2, 128]）匹配，确保能逐元素相加。
            
            first_res_layer = res_layer  ## 临时保存第一层结果（预留，当前未使用）
            
            ret_dicts.append(res_layer)  # # 将当前层结果存入ret_dicts，最终ret_dicts会有多个元素（对应预测头和解码器层的层数）

            # for next level positional embedding
            # 更新下一层的位置嵌入 
                    #res_layer["center"].permute(0, 2, 1)：将绝对中心坐标的形状从[4, 2, 128]调整回[4, 128, 2]
                    #clone()：复制张量，避免后续修改影响原始的res_layer["center"]；
                    #detach()：切断该张量与计算图的连接，确保位置嵌入的更新不会影响之前层的参数训练（仅用预测结果作为下一层的输入，不参与反向传播）。

            #作用：用当前层预测的 “更精准的中心坐标” 作为下一层的初始位置嵌入，实现 “迭代优化”—— 每一层都站在上一层的肩膀上进一步微调，逐步逼近真实目标位置。
                #更新查询位置嵌入  使用优化迭代过的进行输入下一个编码器层中 逐层迭代
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #补充热图相关信息
        #################################
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )                                               # 形状[4, 10, 128]（4个样本，10个类别，128个候选框）
        ret_dicts[0]["dense_heatmap"] = dense_heatmap  
                        #dense_heatmap：原始的稠密热图（未筛选的所有 32400 个位置的分数），存入ret_dicts[0]用于：
                                # 可视化（查看热图分布是否合理）；
                                # 辅助监督（如计算热图回归损失，优化热图生成质量）

        if self.auxiliary is False: #结果返回逻辑：根据模式返回不同结果
                                    #推理阶段（auxiliary=False）：只返回最后一层的结果（ret_dicts[-1]），因为经过 6 层迭代后，最后一层的预测最精准
            # only return the results of last decoder layer
            return [ret_dicts[-1]]  #如果false  讲列表中记录的结果的最后一层返回为最终结果。

        # return all the layer's results for auxiliary superivison #
        ## 辅助监督模式（训练时）：返回所有层的结果
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                 # 拼接所有层的预测参数（如center、size等）
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                # 热图相关键只保留第一层的结果（与层数无关）
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [None], [metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx : batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            list_of_pred_dict,
            np.arange(len(gt_labels_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict["center"].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height, vel
        )  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]["bboxes"]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1), :
            ]
            score_layer = score[
                ...,
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1),
            ]

            if self.train_cfg.assigner.type == "HungarianAssigner3D":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == "HeuristicAssigner":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])
        feature_map_size = (
            grid_size[:2] // self.train_cfg["out_size_factor"]
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            self.num_classes, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
            length = length / voxel_size[1] / self.train_cfg["out_size_factor"]
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                )
                radius = max(self.train_cfg["min_radius"], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.train_cfg["out_size_factor"]
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.train_cfg["out_size_factor"]
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                # NOTE: fix
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, "on_the_image_mask"):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict["dense_heatmap"]),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict["loss_heatmap"] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                idx_layer == 0 and self.auxiliary is False
            ):
                prefix = "layer_-1"
            else:
                prefix = f"layer_{idx_layer}"

            layer_labels = labels[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_label_weights = label_weights[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_score = preds_dict["heatmap"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score,
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict["center"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_height = preds_dict["height"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_rot = preds_dict["rot"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_dim = preds_dict["dim"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot], dim=1
            ).permute(
                0, 2, 1
            )  # [BS, num_proposals, code_size]
            if "vel" in preds_dict.keys():
                layer_vel = preds_dict["vel"][
                    ...,
                    idx_layer
                    * self.num_proposals : (idx_layer + 1)
                    * self.num_proposals,
                ]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1
                ).permute(
                    0, 2, 1
                )  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get("code_weights", None)
            layer_bbox_weights = bbox_weights[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
                code_weights
            )
            layer_bbox_targets = bbox_targets[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ]
            layer_loss_bbox = self.loss_bbox(
                preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1)
            )

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f"{prefix}_loss_cls"] = layer_loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f"matched_ious"] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals :].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(
                self.query_labels, num_classes=self.num_classes
            ).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot

            batch_center = preds_dict[0]["center"][..., -self.num_proposals :]
            batch_height = preds_dict[0]["height"][..., -self.num_proposals :]
            batch_dim = preds_dict[0]["dim"][..., -self.num_proposals :]
            batch_rot = preds_dict[0]["rot"][..., -self.num_proposals :]
            batch_vel = None
            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"][..., -self.num_proposals :]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg["dataset"] == "nuScenes":
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=["pedestrian"],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=["traffic_cone"],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg["dataset"] == "Waymo":
                self.tasks = [
                    dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                    dict(
                        num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                    ),
                    dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]["bboxes"]
                scores = temp[i]["scores"]
                labels = temp[i]["labels"]
                ## adopt circle nms for different categories
                if self.test_cfg["nms_type"] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task["indices"]:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task["radius"] > 0:
                            if self.test_cfg["nms_type"] == "circle":
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task["radius"],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]["box_type_3d"](
                                        boxes3d[task_mask][:, :7], 7
                                    ).bev
                                )
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task["radius"],
                                    pre_maxsize=self.test_cfg["pre_maxsize"],
                                    post_max_size=self.test_cfg["post_maxsize"],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][
                                task_keep_indices
                            ]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [
            [
                metas[0]["box_type_3d"](
                    rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1]
                ),
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]
        return res
