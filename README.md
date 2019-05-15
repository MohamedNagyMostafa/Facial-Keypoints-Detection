<h1>Facial Keypoints Detection</h1>

<p>Using CNN model to recognize the facial keypoints over different faces in the image</p>

<h2>CNN Structure</h2>
<p>The CNN structure has a unique structure that contains five convolution layers with composed <b>Pulling Convolution Layers</b> for compasing features.</p>

<br />
<img src="https://user-images.githubusercontent.com/20774864/57783549-9c89fa80-772e-11e9-9df2-d30e3ed40d34.png"/>
<br />

<h2>Data Preprocessing</h2>
<ul>
<li>Normalizing the image to <b>binary image</b> and keypoints in <b>range -1~1</b></li>
<li>Rescaling the images to 224x224</li>
<li>Random crop</li>
<li>Random rotation about 15 degres </li>
</ul>

<h2>Optimizer & Loss Function</h2>
<p>Using Adam optimizer and <b>Smooth L1 Loss</b> as a regression loss function.
<br />
<img src="https://cdn-images-1.medium.com/max/1000/1*ct5e8rEJYIK4SJxPTYEFWA.png"/>
<br />

<h2>Recognize Facial Keypoints Over Image</h2>
<br/>
<br/>
<img src="https://user-images.githubusercontent.com/20774864/57785733-98f87280-7732-11e9-9795-9d690aafee9c.png"/>
<br />
<br />
