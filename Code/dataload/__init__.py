from dataload.strand_loader import strand_loader

from dataload.image_loader import image_loader
from dataload.imageCal_loader import imageCal_loader
from dataload.step_loader import step_loader
from dataload.origin_step_loader import origin_step_loader
import torch
def data_loader(opt):
    if opt.data_mode=='strand':
        instance=strand_loader()
        instance.initialize(opt)
        return create_dataloader(opt,instance)

    if opt.data_mode=='image':
        instance=image_loader()
        instance.initialize(opt)
        return create_dataloader(opt,instance)
    if opt.data_mode=='imageCal':
        instance=imageCal_loader()
        instance.initialize(opt)
        return create_dataloader(opt,instance)
    if opt.data_mode=='step':
        instance=origin_step_loader()
        instance.initialize(opt)
        return create_dataloader(opt,instance)




def create_dataloader(opt,instance):
    if opt.isTrain:
        print("dataset [%s] of size %d was created" %
              (type(instance).__name__, len(instance)))
        dataloader = torch.utils.data.DataLoader(
            instance,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.nThreads),
            drop_last=opt.isTrain
        )
        return dataloader
    else:
        return instance