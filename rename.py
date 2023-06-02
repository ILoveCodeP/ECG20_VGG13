from tensorflow.keras.models import load_model
import pickle
train_list = [0.5,0.9,0,1.3,1.5,1.9,1,2,5.5,6,'mixwithout0.5','mix']
num_filter_blocks=[64]
nb_cnns = [3,4,5]
# for alpha_train in train_list:
#     for repeat_time in range(5):
#         for num_filter_block in num_filter_blocks:
#             for nb_cnn in nb_cnns:
#                 model = load_model('./models/vgg_trainalpha{}_repeattimes{}_numfilters{}_num_blocks{}.h5'.format(alpha_train,repeat_time,num_filter_block,nb_cnn))
#                 model.save('./model/vgg_trainalpha{}_repeattimes{}_numfilters{}_num_blocks{}.h5'.format(alpha_train,repeat_time,num_filter_block,nb_cnn))

for alpha_train in train_list:
    for repeat_time in range(5):
        for num_filter_block in num_filter_blocks:
            for nb_cnn in nb_cnns:
                df=open('./result/ECG20_VGG13_numfilters{}_b256_alpha{}_sigma0.03_num_blocks{}.txt'.format(num_filter_block,alpha_train,nb_cnn),'rb')#注意此处是rb
                #此处使用的是load(目标文件)
                temp=pickle.load(df)
                df.close()
                df2=open('./results/ECG20_VGG13_numfilters{}_b256_alpha{}_sigma0.03_num_blocks{}.txt'.format(num_filter_block,alpha_train,nb_cnn),'wb')# 注意一定要写明是wb 而不是w.
                #最关键的是这步，将内容装入打开的文件之中（内容，目标文件）
                pickle.dump(temp,df2)
                df2.close()