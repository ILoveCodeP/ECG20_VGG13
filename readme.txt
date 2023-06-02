train.py和test.py都是用[0,2,1.9,1.5,1.3,1,0.9,0.5,multiple,mixture]训练和测试的结果，multiple和mixture包括0.5-2全部的noise
only_multi:多训练一组multiple,其中不包括alpha=0.5的时候,测试的时候用的multiple和mixture包括0.5
only_mix：多训练一组mixture，其中不包括alpha=0.5，测试的时候用的multiple和mixture包括0.5