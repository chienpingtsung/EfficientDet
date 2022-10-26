import logging

import torch

logger = logging.getLogger(__name__)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'{torch.cuda.device_count()} cuda device available.')
    logger.info(f'Using the {device} device.')
    return device
