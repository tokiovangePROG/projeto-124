# Para capturar os quadros
import cv2

# Para processar o array de imagens
import numpy as np


# importe os módulos tensorflow e carregue o modelo
import tensorflow as tf

# Anexando a câmera indexada como 0, com o software da aplicação
camera = cv2.VideoCapture(0)

mymodel = tf.keras.models.load_model('keras_model.h5')
# Loop infinito
while True:

	# Lendo / requisitando um quadro da câmera 
	status , frame = camera.read()

	# Se tivemos sucesso ao ler o quadro
	if status:

		# Inverta o quadro
		frame = cv2.flip(frame , 1)
		
		
				
		# Redimensione o quadro
		resized_frame = cv2.resize(frame , (224,224))
		# Expanda a dimensão do array junto com o eixo 0
		resized_frame = np.expand_dims(resized_frame , axis = 0)
		# Normalize para facilitar o processamento
		resized_frame = resized_frame / 255
		# Obtenha previsões do modelo
		predictions = mymodel.predict(resized_frame)
		

		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)
		# Exibindo os quadros capturados
		cv2.imshow('feed' , frame)


		print(f"Pedra: {rock} %, Papel: {paper} %, Tesoura: {scissor} %")
		# Aguardando 1ms
		code = cv2.waitKey(1)
		
		# Se a barra de espaço foi pressionada, interrompa o loop
		if code == 32:
			break

# Libere a câmera do software da aplicação
camera.release()

# Feche a janela aberta
cv2.destroyAllWindows()
