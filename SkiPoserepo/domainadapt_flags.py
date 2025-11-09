from absl import flags
import os


FLAGS = flags.FLAGS

# DIRECTORIES
flags.DEFINE_string('data_path','/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski','datasets main directory')
flags.DEFINE_string('dataset', "Ski_2DPose_Dataset", 'Dataset name. Choose btw: Ski_2DPose_Dataset, youtube_skijump_dataset')
flags.DEFINE_string('load_checkpoint','','Checkpoint path, the same for training result and for demo mode')

# target
flags.DEFINE_integer('n_joints', 25, 'Number of target joints. 24,25,17')
# flags.DEFINE_bool('aligned_skis',True,'Straight skis or not')

## Image Embedded Parameters
flags.DEFINE_integer('img_size',224,'Image Size')
flags.DEFINE_integer('patch_size',40,'Image Size')
flags.DEFINE_integer('in_chans',3,'input channels')
flags.DEFINE_integer('mask_size',32,'mask size')
flags.DEFINE_float('mask_ratio',0.6,'mask ratio')
# flags.DEFINE_integer('drop_rate',0.1,'drop rate')


# Training Parameters
flags.DEFINE_bool('freeze_fe',False,'Freeze feature extractor')
flags.DEFINE_bool('use_backbone',False,'Use Rsnet or not')
flags.DEFINE_bool('use_image_patches',True,'Use Image Patches')
flags.DEFINE_float('lr',2e-4, 'Learning Rate')
flags.DEFINE_float('b1', 0.5, 'Beta1')
flags.DEFINE_float('b2', 0.999, 'Beta2')
flags.DEFINE_integer('num_workers', int(os.cpu_count() / 2), 'Number of workers.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('n_epochs', 25, 'Number of training epochs.')
# flags.DEFINE_string('masking_mode','zero','Choose masking mode. ski, random')
flags.DEFINE_integer('masked_joints', 6, 'Number of masked joints: 13,14,6, (only for random masking mode)')

# flags.DEFINE_bool('data_augmentation', False, 'Simmetry and rotation of training data')
# flags.DEFINE_bool('weighted_loss', False, 'Weighted loss (based on the source dataset)')
# flags.DEFINE_bool('fine_tuning',True,'Fine tuning a pretrained network on ski jump dataset')
# flags.DEFINE_string('load_pretrained_model','/home/federico.diprima/DAT/DAMski/checkpoints/8batch_Ski2D.ckpt','Checkpoint path used for finetuning')
# flags.DEFINE_string('train_dataset','ski','Choose train dataset (target), always ski')
# flags.DEFINE_string('train_set_mode','S2D_SJ','Choose what ski set use for training (S2D, SJ, S2D_SJ, S2D_SJ+S)')
# flags.DEFINE_integer('testing_jump', 4, 'Choose the jump on which the train and test is performed (1,2,3,4)')

# Filter openpose bad estimation
# flags.DEFINE_string('filter_openpose',"delete_pose",'Filter bad joints estimated from openpose (no_filter, delete_pose')

#Mode
flags.DEFINE_string('mode','train','Train Mode (train or demo)')
