/*
 * Copyright (C) 2019 Elias Yilma
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package cnn;

import UTIL.Mat;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 *
 * @author Elias Yilma
 * 
 * A Simple Convolutional Neural Network based on the tutorial by Victor Zhou at
 * https://victorzhou.com/blog/intro-to-cnns-part-1/
 * https://victorzhou.com/blog/intro-to-cnns-part-2/
 * 
 * The network classifies digits from the MNIST digits database with an average 
 * accuracy of about 90%. 
 * 
 * The CNN consists of four layers namely: 
 *          -the convolution layer (Convolution.java)
 *          -the maximum pooling layer (MaxPool.java)
 *          -the softmax activation layer (SoftMax.java)
 *          -the output layer that implements cross-entropy loss.
 * 
 * 
 * 
 */
public class CNN {

    /** Loads image from file and returns a bufferedImage.
     * @param src Absolute file path to image
     * @return BufferedImage loaded from file. 
     * @throws java.io.IOException 
    **/
    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }


    /**
     * converts a BufferedImage into a pixel array and normalizes it. 
     * @param imageToPixelate the source image to be converted.
     * @return 2D array with normalized pixel values between 0.0 and 1.0
     */
    public static float[][] img_to_mat(BufferedImage imageToPixelate) {
        int w = imageToPixelate.getWidth(), h = imageToPixelate.getHeight();
        int[] pixels = imageToPixelate.getRGB(0, 0, w, h, null, 0, w);
        float[][] dta = new float[w][h];

        for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel++) {
            dta[row][col] = (((int) pixels[pixel] >> 16 & 0xff)) / 255.0f;
            col++;
            if (col == w) {
                col = 0;
                row++;
            }
        }
        return dta;
    }

    /**
     * creates 3X3 convolution filters with random initial weights.
     * @param size number of 3X3 filters to be randomly initialized
     * @return a [size] X [3] X [3] 3d array with size filters
     */
    public static float[][][] init_filters(int size) {
        float[][][] result = new float[size][3][3];
        for (int k = 0; k < size; k++) {
            result[k] = Mat.m_random(3, 3);
        }
        return result;
    }

    /**
     * loads a random image from a specific digit folder in the MNIST database.
     * @param label the folder label (ranges between 0 and 9)
     * @return a BufferedImage of the digit.
     * @throws IOException if the image file isn't found.
     */
    public static BufferedImage mnist_load_random(int label) throws IOException {
        String mnist_path = "data\\mnist_png\\mnist_png\\training";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "\\" + label + "\\" + files[random_index];
        BufferedImage bi = load_image(final_path);
        return bi;
    }
    
    /**
     * performs both the forward and back-propagation passes of the CNN.
     * @param training_size the number of images used for training the CNN.
     * @throws IOException if image cannot be found.
     */
    public static void train(int training_size) throws IOException {
        float[][][] filters = init_filters(8);
        int label_counter = 0;
        float ce_loss=0;
        int accuracy=0;
        float acc_sum=0.0f;
        float learn_rate=0.005f;
        
        //initialize layers
        Convolution conv=new Convolution();
        MaxPool pool=new MaxPool();
        SoftMax softmax=new SoftMax(13*13*8,10);

        float[][] out_l = new float[1][10];    
        for (int i = 0; i < training_size; i++) {
            //grab a random image from database.
            BufferedImage bi = mnist_load_random(label_counter);
            int correct_label = label_counter;
            if(label_counter==9){
                label_counter=0;
            }else{
                label_counter++;
            }
            
            //FORWARD PROPAGATION
            
            //convert to pixel array
            float[][] pxl = img_to_mat(bi);
            // perform convolution 28*28 --> 8x26x26
            float[][][] out = conv.forward(pxl, filters);

            // perform maximum pooling  8x26x26 --> 8x13x13
            out = pool.forward(out);
            
            // perform softmax operation  8*13*13 --> 10
            out_l = softmax.forward(out); 
            
            // compute cross-entropy loss
            ce_loss += (float) -Math.log(out_l[0][correct_label]);
            accuracy += correct_label == Mat.v_argmax(out_l) ? 1 : 0;
            
            //BACKWARD PROPAGATION --- STOCHASTIC GRADIENT DESCENT
            //gradient of the cross entropy loss
            float[][] gradient=Mat.v_zeros(10);
            gradient[0][correct_label]=-1/out_l[0][correct_label];
            float[][][] sm_gradient=softmax.backprop(gradient,learn_rate);
            float[][][] mp_gradient=pool.backprop(sm_gradient);
            conv.backprop(mp_gradient, learn_rate);
            if(i % 100 == 99){
                System.out.println(" step: "+ i+ " loss: "+ce_loss/100.0+" accuracy: "+accuracy);
                ce_loss=0;
                acc_sum+=accuracy;
                accuracy=0;
            }
        }
        System.out.println("average accuracy:- "+acc_sum/training_size+"%");
    }

    
      
    /**
     * Test method.
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {      
        train(30000);
    }

}
