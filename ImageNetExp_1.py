# Get the explanation of ImageNet 
from multiprocessing import Process
from methods.Lime import *
from methods.LEGv0 import *
from methods.KernelSHAP import *
from methods.CShap import *
from methods.GradCam import *
def make_job(method, spsz, gpu_id):
    if gpu_id > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    #Read the whole data:
    Num = 500
    VGG19_MODEL = VGG19(include_top = True)
    for i in range(Num):
        img = image.load_img('/CCAS/home/shine_lsy/LEG/imagenet/images/'+str(i)+'.png', target_size=(224,224))
        img = image.img_to_array(img).astype(int)
        image_input = np.expand_dims(img.copy(), axis = 0)
        image_input = preprocess_input(image_input)
        preds = VGG19_MODEL.predict(image_input)
        chosen_class = np.argmax(preds)
        
        start = time()
        #LIME
        if method == "Lime":
            sample_size = spsz
            if os.path.isfile("imagenet/result/Lime/"+str(i)+'.txt'):
                print ("File "+str(i)+" exist")
            else:
                print("LIME"+str(i)+" Computing")
                result = Get_Lime(image_input, VGG19_MODEL, sample_size ,chosen_class )
                np.savetxt("imagenet/result/Lime/"+str(i)+'.txt', result)
            end = time()
            print("LIME"+str(i)+" takes "+str(round((end-start)/60, 2))+" mins")
        #LEGv0
        elif method == "LEGv0":
            sample_size = int(spsz/3)
            if os.path.isfile("imagenet/result/LEGv0/"+str(i)+'.txt'):
                print ("LEGv0 File "+str(i)+" exist")
            else:
                print("LEGv0 "+str(i)+" computing")
               
                result = LEG_explainer(np.expand_dims(img.copy(), axis = 0), VGG19_MODEL, predict_vgg19,base_size = 28, noise_lvl = 0.02, num_sample = sample_size, penalty='TV',lambda_arr = [0.1])
                np.savetxt("imagenet/result/LEGv0/"+str(i)+'.txt',result[0][0])
        #KernelSHAP
        elif method == "KernelSHAP": 
            sample_size = spsz
            if os.path.isfile("imagenet/result/KernelSHAP/"+str(i)+'.txt'):
                print ("KernelSHAP File "+str(i)+" exist")
            else:
                print("KernelSHAP "+str(i)+" computing")
                
                result = Get_KernelSHAP(img, VGG19_MODEL, sample_size, chosen_class)
                np.savetxt("imagenet/result/KernelSHAP/"+str(i)+'.txt',result)
        #CShap
        elif method == "CShap":
            sample_size = spsz
            if os.path.isfile("imagenet/result/CShap/"+str(i)+'.txt'):
                print ("CShap File "+str(i)+" exist")
            else:
                print("CShap "+str(i)+" computing")
                result = Get_CShap(img, VGG19_MODEL, chosen_class, max_order=8)
                np.savetxt("imagenet/result/CShap/"+str(i)+'.txt',result)
        #GradCam
        elif method == "GradCam":
            sample_size = spsz
            if os.path.isfile("imagenet/result/GradCam/"+str(i)+'.txt'):
                print ("File "+str(i)+" exist")
            else:
                print("GradCam "+str(i)+" computing")
                result = Get_GradCam(image_input, VGG19_MODEL)
                np.savetxt("imagenet/result/GradCam/"+str(i)+'.txt',result)
        #LEG0
        elif method == "LEG0": 
            sample_size = int(spsz/3)
            if os.path.isfile("imagenet/result/LEG/"+str(i)+'.txt'):
                print ("LEG File "+str(i)+" exist")
            else:
                print("LEG "+str(i)+" computing") 
                result = LEG_explainer(np.expand_dims(img.copy(), axis = 0), VGG19_MODEL, predict_vgg19,base_size = 28, num_sample = sample_size, penalty=None, noise_lvl = 0.02)
                np.savetxt("imagenet/result/LEG/"+str(i)+'.txt',result[0][0])
        else:
            return("None")
        end = time()
        print(method+str(i)+" takes "+str(round((end-start)/60, 2))+" mins")
    return(method+" Completed")




if __name__ == "__main__":
    p1 = Process(target = make_job, args = ("KernelSHAP",6000 ,0))
    p2 = Process(target = make_job, args = ("CShap",6000 ,1))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
   # p3 = Process(target = make_job, args = ("LEGv0",6000 ,2)) 
   # p3.start()
   # p3.join()
   # p4 = Process(target = make_job, args = ("Lime",6000 ,3))
   # p4.start()
   # p4.join() 
   # p1 = Process(target = make_job, args = ("LIME",1000 ,0)) 
   # p1 = Process(target = make_job, args = ("LIME",1000 ,0)) 
   
   
   
   







