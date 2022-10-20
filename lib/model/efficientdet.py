from torch import nn

from lib.block.efficientdet import BiFPN, FPNStem, Classifier, Regressor, Anchors
from lib.model.efficientnetv2 import EfficientNetV2


class EfficientDet(nn.Module):
    @staticmethod
    def model_args(compound_coef):
        pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        fpn_channels = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        fpn_layers = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        head_layers = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        anchor_scale = [4, 4, 4, 4, 4, 4, 4, 5, 4]
        return {'pyramid_levels': pyramid_levels[compound_coef],
                'fpn_channels': fpn_channels[compound_coef],
                'fpn_layers': fpn_layers[compound_coef],
                'fpn_attention': True if compound_coef < 6 else False,
                'head_layers': head_layers[compound_coef],
                'anchor_scale': anchor_scale[compound_coef]}

    def __init__(self, num_classes, scales, ratios, levels, scale_feat, i_c=3, compound_coef=0, scale='xs'):
        super(EfficientDet, self).__init__()

        self.scale_feat = scale_feat

        args = EfficientDet.model_args(compound_coef)
        pyramid_levels = args['pyramid_levels']
        fpn_channels = args['fpn_channels']
        fpn_layers = args['fpn_layers']
        fpn_attention = args['fpn_attention']
        head_layers = args['head_layers']
        anchor_scale = args['anchor_scale']
        args = EfficientNetV2.model_args(scale)
        scale_channels = args['scale_channels']
        bn_eps = args['bn_eps']
        bn_mom = args['bn_mom']

        self.backbone = EfficientNetV2(i_c, scale=scale)
        del self.backbone.head
        self.fpnstem = FPNStem(scale_channels[-self.scale_feat:], fpn_channels, pyramid_levels, bn_eps, bn_mom)
        self.fpn = nn.Sequential(*[BiFPN(fpn_channels, bn_eps, bn_mom, pyramid_levels, fpn_attention)
                                   for _ in range(fpn_layers)])
        num_anch = len(scales) * len(ratios)
        self.classifier = Classifier(fpn_channels, num_anch, num_classes, head_layers, pyramid_levels, bn_eps, bn_mom)
        self.regressor = Regressor(fpn_channels, num_anch, head_layers, pyramid_levels, bn_eps, bn_mom)
        self.anchors = Anchors(anchor_scale, levels, scales, ratios)

    def forward(self, x):
        anchors = self.anchors(x)

        x = self.backbone.forward_backbone(x)
        x = self.fpnstem(x[-self.scale_feat:])
        x = self.fpn(x)

        classification = self.classifier(x)
        regression = self.regressor(x)

        return classification, regression, anchors
