import torch
import torch.optim as optim
from model import DSMCR_NN
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label


# Start running

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # data parameters
    DATA_DIR = 'data/pascal/'

    # hyper-parameters
    # pascal
    alpha = 1e-3
    beta = 1e-2

    # #wiki
    # alpha = 1e-3
    # beta = 1e-2


    MAX_EPOCH = 20
    batch_size = 32
    lr = 1e-5
    betas = (0.5, 0.999)
    weight_decay = 0

    print('...Data loading is beginning...')
    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
    print('...Data loading is completed...')
    #print(input_data_par['img_dim'])
    #print(input_data_par['text_dim'])
    model_ft = DSMCR_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], output_dim=input_data_par['num_class']).to(device)
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)

    print('...Training is beginning...')
    # Train and evaluate
    model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta, MAX_EPOCH)
    print('...Training is completed...')

    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict = model_ft(torch.tensor(input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))
    label = torch.argmax(torch.tensor(input_data_par['label_test']), dim=1)
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))
    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
