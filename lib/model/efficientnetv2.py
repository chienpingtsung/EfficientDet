from torch import nn

from lib.block.efficientnetv2 import Stem, FusedMBConvBlock, MBConvBlock, Head


class EfficientNetV2(nn.Module):
    @staticmethod
    def model_args(scale):
        block_args = {'xs': {'dropout': 0.2,
                             'block': [('f', 1, 3, 1, 1, 32, 16, None),
                                       ('f', 2, 3, 2, 4, 16, 32, None),
                                       ('f', 2, 3, 2, 4, 32, 48, None),
                                       ('v', 3, 3, 2, 4, 48, 96, 0.25),
                                       ('v', 5, 3, 1, 6, 96, 112, 0.25),
                                       ('v', 8, 3, 2, 6, 112, 192, 0.25)]},
                      's': {'dropout': 0.2,
                            'block': [('f', 2, 3, 1, 1, 24, 24, None),
                                      ('f', 4, 3, 2, 4, 24, 48, None),
                                      ('f', 4, 3, 2, 4, 48, 64, None),
                                      ('v', 6, 3, 2, 4, 64, 128, 0.25),
                                      ('v', 9, 3, 1, 6, 128, 160, 0.25),
                                      ('v', 15, 3, 2, 6, 160, 256, 0.25)]},
                      'm': {'dropout': 0.3,
                            'block': [('f', 3, 3, 1, 1, 24, 24, None),
                                      ('f', 5, 3, 2, 4, 24, 48, None),
                                      ('f', 5, 3, 2, 4, 48, 80, None),
                                      ('v', 7, 3, 2, 4, 80, 160, 0.25),
                                      ('v', 14, 3, 1, 6, 160, 176, 0.25),
                                      ('v', 18, 3, 2, 6, 176, 304, 0.25),
                                      ('v', 5, 3, 1, 6, 304, 512, 0.25)]},
                      'l': {'dropout': 0.4,
                            'block': [('f', 4, 3, 1, 1, 32, 32, None),
                                      ('f', 7, 3, 2, 4, 32, 64, None),
                                      ('f', 7, 3, 2, 4, 64, 96, None),
                                      ('v', 10, 3, 2, 4, 96, 192, 0.25),
                                      ('v', 19, 3, 1, 6, 192, 224, 0.25),
                                      ('v', 25, 3, 2, 6, 224, 384, 0.25),
                                      ('v', 7, 3, 1, 6, 384, 640, 0.25)]},
                      'xl': {'dropout': 0.4,
                             'block': [('f', 4, 3, 1, 1, 32, 32, None),
                                       ('f', 8, 3, 2, 4, 32, 64, None),
                                       ('f', 8, 3, 2, 4, 64, 96, None),
                                       ('v', 16, 3, 2, 4, 96, 192, 0.25),
                                       ('v', 24, 3, 1, 6, 192, 256, 0.25),
                                       ('v', 32, 3, 2, 6, 256, 512, 0.25),
                                       ('v', 8, 3, 1, 6, 512, 640, 0.25)]}}
        block_args = block_args[scale]
        (*_, block_i_c, _, _), *_, (*_, block_o_c, _) = block_args['block']
        return {**block_args,
                'block_io_c': (block_i_c, block_o_c),
                'num_blocks': sum(r for _, r, *_ in block_args['block']),
                'bn_eps': 1e-3,
                'bn_mom': 1e-2,
                'sd_ratio': 0.2,
                'head_conv_o_c': 1280}

    def __init__(self, i_c=3, num_classes=1000, scale='xs'):
        super(EfficientNetV2, self).__init__()

        args = EfficientNetV2.model_args(scale)
        block_i_c, block_o_c = args['block_io_c']
        num_blocks = args['num_blocks']
        bn_eps = args['bn_eps']
        bn_mom = args['bn_mom']
        sd_ratio = args['sd_ratio']

        self.stem = Stem(i_c, block_i_c, 3, 2, bn_eps, bn_mom)

        self.block = nn.ModuleList()
        n_b = 0
        for t, r, k, s, e, i, o, se in args['block']:
            if t == 'f':
                self.block.append(FusedMBConvBlock(i, o, k, s, bn_eps, bn_mom, e, sd_ratio * n_b / num_blocks))
                n_b += 1
                for _ in range(r - 1):
                    self.block.append(FusedMBConvBlock(o, o, k, 1, bn_eps, bn_mom, e, sd_ratio * n_b / num_blocks))
                    n_b += 1
            elif t == 'v':
                self.block.append(MBConvBlock(i, o, k, s, bn_eps, bn_mom, e, se, sd_ratio * n_b / num_blocks))
                n_b += 1
                for _ in range(r - 1):
                    self.block.append(MBConvBlock(o, o, k, 1, bn_eps, bn_mom, e, se, sd_ratio * n_b / num_blocks))
                    n_b += 1

        self.head = Head(block_o_c, args['head_conv_o_c'], bn_eps, bn_mom, args['dropout'], num_classes)

    def forward_backbone(self, x):
        x = self.stem(x)

        scale_features = []
        for block in self.block:
            x0 = x
            x = block(x)
            if x0.shape[2:] != x.shape[2:]:
                scale_features.append(x0)
        scale_features.append(x)

        return scale_features

    def forward(self, x):
        *_, x = self.forward_backbone(x)

        x = self.head(x)

        return x
