# Image-inpainting-project
Deep learning applications have been evolving rapidly and tremendously as various techniques emerge to fulfill image processing tasks. Image Inpainting being one of these techniques holds an important role for image restoration and content gener- ation. This project will therefore reimplement and enhance an Image Inpainting algorithm proposed by Liu et al.[3]. Our primary goal is to achieve the results proposed in the paper using the proposed methodology and a different dataset. We reimplement the Partial Convolution approach that specifically addresses the reconstruction of images with irregular line masks while preserving its semantic context. Partial Convolutions comprise two main components: the encoder and the decoder. The encoder is adept at leveraging the information from unmasked pixels around the masked areas, ensuring the inpainting process is both seamless and contextually relevant. On the other hand, the decoder is responsible for creating the content within any random region of the image. When training Partial Convolution, we have experimented with three different loss: a standard pixel-wise L2 loss, an adversarial loss proposed by Pathak et al.[4], and a perceptual loss. The adversarial loss produces much clear results and the perceptual loss minimizes the difference between the inpainted and original images by utilizing VGG pre-trained CNN. Be- sides, we also utilize Gaussian noise to improve the robustness and generalization of our neural network. We systematically analyze the model’s performance through extensive experiments, demonstrating its advantage over conventional methods in handling complex irregular line masks.

## Our final report
[Image inpainting report](https://github.com/Cathyyyyy123/Image-inpainting-project/blob/main/Project_Report.pdf)

## Contributors
- Chang Shu
- Burak Sahinkaya