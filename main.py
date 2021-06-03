import func as f
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = f.load_image_and_label()
model = f.make_model()

last_conv_layer_name = 'conv2d'
img = train_images[0]
img = img.reshape(-1, 32, 32, 3)
preds = model.predict(img)
heatmap = f.make_gradcam_heatmap(img, model, last_conv_layer_name)
plt.matshow(heatmap)

history = model.fit(train_images, train_labels, epochs=10, validation_split=0.25)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("loss: ", test_loss, " acc: ", test_acc)

preds = model.predict(img)
heatmap = f.make_gradcam_heatmap(img, model, last_conv_layer_name)
plt.matshow(heatmap)

plt.show()