class args():
    # training args
    epochs = 9  #"number of training epochs, default is 2"
    batch_size =8 #"batch size for training, default is 4"
    test_batch = 1

    dataset_ir = "./m3fd_all/data/ir_new"
    dataset_vis = "./m3fd_all/data/vis_new"
    dataset_test_ir = './test_data/ir' #'./test_images/ir'
    dataset_test_vis ='./test_data/vis' #'./test_images/vis'
    test_output = "./output/H_images"
    seed = 2023
    dataset = "/root/autodl-tmp/DDNGAN/pretrain1/dataset/vis"
    HEIGHT = 128
    WIDTH = 128
    test_HEIGHT = 128
    test_WIDTH = 128
    save_model_dir_autoencoder = "./models_rgb/nestfuse_autoencoder/"
    save_loss_dir = "./models_rgb/loss_autoencoder/"
    input_channels = 6
    output_channels = 1
    cuda = 1


    pixel_loss_weight = [0.6, 0.7, 0.8, 0.9, 1.0]#像素权重
    grad_loss_weight = [0.6, 0.7, 0.8, 0.9, 1.0]#梯度权重
    ssim_innerweight = [0.6, 0.7, 0.8, 0.9, 1.0]#ssim权重
    nb_filter = [32, 64, 128, 256, 512] #[64, 128, 256, 512, 1024]
    ssim_num = 2#ssim网络层权重补偿
    ssim_weight = [1,10,100,1000,10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']



    #lr_G = 2e-5  #"learning rate, default is 0.001"
    #lr_D = 2e-5  #"learning rate, default is 0.001"
    lr_G = 5e-6
    lr_D = 5e-6
    lambda_term = 10
    is_train = 1
    lr_light = 5e-5  # "learning rate, default is 0.001"
    log_interval = 10  #"number of images after which the training loss is logged, default is 500"
    resume = None# #'/root/autodl-tmp/models_lambda_2b_p0.6_g0.6/nestfuse_autoencoder/Epoch_6_iters_5000_Tue_Jun_13_09_15_29_2023_G.model' #
    resume_G ='./models/ligth/Final_epoch_G16_Fri_Mar_14_01_52_26_2025_.model'
    
    resume_D_i = './models/ligth/Final_epoch_D_i16_Fri_Mar_14_01_52_26_2025_.model'
    resume_D_v = './models/ligth/Final_epoch_D_v16_Fri_Mar_14_01_52_26_2025_.model'
    # for test, model_default is the model used in paper
    model_default = 'Final_epoch_G16_7_Fri_Mar_14_17_21_31_2025_.model'

    model_deepsuper = './models/nestfuse_1e2_deep_super.model'
    model_path_rgb = '/root/autodl-tmp/models_rgb/nestfuse_autoencoder/Final_epoch_G30_Tue_Jan_14_20_45_52_2025_.model'

