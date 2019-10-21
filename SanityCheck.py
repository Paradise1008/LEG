#!/usr/bin/env python
# coding: utf-8

# In[2]:


from LEG import * 
class Image_Obj:
    def __init__(self,name,Noise,Lambda):
        self.name = name
        self.Noise = Noise
        self.Lambda = Lambda
List = []
List.append(Image_Obj("shark",0.3,0.075))
List.append(Image_Obj("soccer",0.3,0.075))
List.append(Image_Obj("cellphone",0.3,0.008))


image_folder = "Image"
vgg_model = vgg19.VGG19(include_top =True)
my_weights = np.array(vgg_model.get_weights()) 
new_weights = my_weights.copy()
for i, val in enumerate(List):
    image0 = image.load_img(os.path.join(image_folder,val.name+'.jpg'), target_size=(224,224))
    image0 = image.img_to_array(image0)
    image_input = np.expand_dims(image0.copy(),axis=0)
    image_input = vgg19.preprocess_input(image_input)
    preds = vgg_model.predict(image_input)
    for x in decode_predictions(preds)[0]:
        print(x)
    chosen_class = np.argmax(preds)
    for k in range(3):
        if k == 0:
            task = LEG_Explain(vgg_model, image0, val.name , np.array([val.Noise]) , np.array([val.Lambda]) ,sampling_size = 200, conv = 8,chosen_class=chosen_class)
            generateHeatmap(image0,task[0].sol,result_path = "SC_Result",name = val.name+'0'+'.jpg',style = "heatmap_only",showOption=True, direction="all")
    
        layer_weight_size = my_weights[37-2*k-1].shape
        layer_bias_size = my_weights[37-2*k].shape
        new_layer_weight = np.random.uniform(low=0.0, high=1.0, size=layer_weight_size)
        new_layer_bias = np.random.uniform(low=0.0, high=1.0, size=layer_bias_size)
        new_weights[37-2*k-1] = new_layer_weight
        new_weights[37-2*k] = new_layer_bias
        ###########update the model###############
        vgg_model.set_weights(new_weights)  
        task = LEG_Explain(vgg_model, image0, val.name , np.array([val.Noise]) , np.array([val.Lambda]) ,sampling_size = 200, conv = 8,chosen_class=chosen_class)
        generateHeatmap(image0,task[0].sol,result_path = "SC_Result",name = val.name+str(k)+'.jpg',style = "heatmap_only",showOption=True, direction="all")
    
print("Completed")

