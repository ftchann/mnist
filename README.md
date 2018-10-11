# mnist
<h1>How to use the Neural Network</h1>
<h2>Train Neural Network</h2>
<p>To train the neural network you need to run either the neuronaleNetzwerke.py or neuronalesNetzwerktrainieren.py.<br> You can change the parameters on top. The best weights and biases will be saved to a .npy. <br>
You can change the of the file to which it saves in lines 295 - 315.<br>
Make sure that you don't loose the pretrained weights. Put them into another folder so that they don't get overwritten.</p>
<h2>Test Neural Network</h2>
<p>To test a neural network you need to run neuronaleNetzwerketesten.py.<br>
  Change the parameters so that your neural network has the same parameters as the parameters from your best weights or biases.<br>
  Load your .npy file and begin the test.<br>
  It will give you what performance you got.</p>
<h2>Ask Neural Network</h2>
<p>To ask the neural network you need run neuronaleNetzwerkeabfragen.py<br>
  With this Programm you can check your own handwriting. Just make a picture from digit.<br>
  The neural network works best if you write with black and the background is white.<br>
  Make sure that the line width is not to small.<br>
  The get the best performance, your digit should be in the same format as the digits in the Mnist dataset.<br>
  Then change to path to your picture. <br>
  Make sure that the parameters from your neuralnetwork are the same as the parameters form your best weights or biases.<br>
  Lead your .npy file and ask the neural network.<br>
  The programm will print you what digit you wrote.<br>
</p>
