##Import the required packages
import sys
sys.path.append('../')
from LEG import * 

###########Reproduce the result in the paper:
class Image_Obj:
    def __init__(self,name,Noise,Lambda):
        self.name = name
        self.Noise = Noise
        self.Lambda = Lambda
List = []
List.append(Image_Obj("soccer",0.3,0.0001))
List.append(Image_Obj("soccer",0.3,0.075))


image_folder = "data"
vgg_model = vgg19.VGG19(include_top =True)
for i, val in enumerate(List):
    image0 = image.load_img(os.path.join(image_folder,val.name+'.jpg'), target_size=(224,224))
    image0 = image.img_to_array(image0)
    image_input = np.expand_dims(image0.copy(),axis=0)
    image_input = vgg19.preprocess_input(image_input)
    preds = vgg_model.predict(image_input)
    for x in decode_predictions(preds)[0]:
        print(x)
    chosen_class = np.argmax(preds)        
    task = LEG_Explain(vgg_model, image0, val.name , np.array([val.Noise]) , np.array([val.Lambda]) ,sampling_size = 200, conv = 8,chosen_class=chosen_class)
    y_vec = task[0].sol.flatten()
    filename = val.name+str(task[0].Lambda)+"LEG.csv"   #First 01 means smooth  Second 01 means take abs
    np.savetxt(filename),y_vec,delimiter=',')
    #generateHeatmap(image0,task[0].sol,result_path="Result",name = val.name+'.jpg',style = "heatmap_only",showOption=True, direction="all")
    #generateHeatmap(image0,task[0].sol,result_path="Result",name = val.name+'_gray.jpg',style = "gray",showOption=True, direction="all")
    
print("Completed")
