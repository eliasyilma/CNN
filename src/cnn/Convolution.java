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

/**
 *
 * @author Elias Yilma
 *
 * The Convolution layer of the neural network.
 * Performs the convolution operation using different filters without padding. 
 * 
 */
public class Convolution {
    //

    /**
     * caches the input data (the image) for use in the back-propagation phase.
     */
        public float[][] input; // shape --> [28] X [28]
    //

    /**
     * caches filters that were used in the convolution phase for use in the 
     * back-propagation phase.
     */
        public float[][][] filters; // shape --> [3] X [8] X [8]
    /**
     * Convolves the image with respect to a 3X3 filter
     * @param image the image matrix with shape [28] X [28]
     * @param filter a 3X3 filter used in the convolution process.
     * @return a 2D matrix with shape [26] X [26].
     */
    public float[][] convolve3x3(float[][] image, float[][] filter) {
        input=image;
        float[][] result = new float[image.length - 2][image[0].length - 2];
        //loop through
        for (int i = 1; i < image.length - 2; i++) {
            for (int j = 1; j < image[0].length - 2; j++) {
                float[][] conv_region = Mat.m_sub(image, i - 1, i + 1, j - 1, j + 1);
                result[i][j] = Mat.mm_elsum(conv_region, filter);
            }
        }
        return result;
    }

    /**
     * the forward convolution pass that convolves the image w.r.t. each filter
     * in the filter array. No padding has been used in this case, so output matrix
     * shape decreases by 2 w.r.t row width and column height.
     * @param image the input image matrix. [28] X [28]
     * @param filter a 3D matrix containing an array of 3X3 filters ([8]X[3]X[3])
     * @return a 3D array containing an array of the convolved images w.r.t.
     * each filter. [8] X [26] X [26]
     */
    public float[][][] forward(float[][] image, float[][][] filter) {
        filters=filter; // 8 X 3 X 3
        float[][][] result = new float[8][26][26];
        for (int k = 0; k < filters.length; k++) {
            float[][] res = convolve3x3(image, filters[k]);
            result[k] = res;
        }
        return result;
    }
    
    /**
     * 
     * @param d_L_d_out the input gradient matrix retrieved from the back-propagation
     *  phase of the maximum pooling stage. shape = [8] X [26] X [26]
     * @param learning_rate the learning rate factor used in the neural network.
     */
    public void backprop(float[][][] d_L_d_out,float learning_rate){
        //the output gradient which is dL/dfilter= (dL/dout)*(dout/dfilter)
        float[][][] d_L_d_filters= new float[filters.length][filters[0].length][filters[0][0].length];
        //reverses the convolution phase by creating a 3X3 gradient filter 
        //and assigning its elements with the input gradient values scaled by
        //the corresponding pixels of the image.
        for(int i=1;i<input.length-2;i++){
            for(int j=1;j<input[0].length-2;j++){
                for(int k=0;k<filters.length;k++){
                    //get a 3X3 region of the matrix
                    float[][] region=Mat.m_sub(input,  i - 1, i + 1, j - 1, j + 1);
                    //for each 3X3 region in the input image i,j
                    // d_L_d_filter(kth filter) = d_L_d_filter(kth filter)+ d_L_d_out(k,i,j)* sub_image(3,3)i,j
                    //       [3] X [3]          =       [3] X [3]         +     gradient    *      [3] X [3]
                    //see article as to how this gradient is computed.
                    d_L_d_filters[k]=Mat.mm_add(d_L_d_filters[k], Mat.m_scale(region,d_L_d_out[k][i-1][j-1]));
                }
            }
        }
        
        //update the filter matrix with the gradient matrix obtained above.
        for(int m=0;m<filters.length;m++){
          // [3] X [3]  =   [3] X [3] + -lr * [3] X [3]   
            filters[m]= Mat.mm_add(filters[m], Mat.m_scale(d_L_d_filters[m],-learning_rate));
        }  
    }
}
