/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package UTIL;
/**
 * MATRIX UTILITIES FOR CNN OPERATIONS
 * 
 * All matrices are represented as 2D and 3D arrays.
 * 
 * All row vectors, unless otherwise stated, are represented as 2D arrays
 * with the shape of [1] X [n].
 * 
 * All column vectors, unless otherwise stated, are represented as 2D arrays
 * with the shape of [m] X [1].
 * 
 * This was done to facilitate matrix operations between matrices and vectors
 * without the added hassle of creating two separate methods or an extra dimension.
 * 
**/
public class Mat {
 
    /**
     * performs naive matrix multiplication between two matrices of shape [k][l] and [l][m]
     * @param m1 a 2D array shape of [k][l]
     * @param m2 a 2D array shape of [l][m]
     * @return a 2D array of shape [k][m]
     */
    public static float[][] mm_mult(float[][] m1, float[][] m2) {
        float[][] result = new float[m1.length][m2[0].length];
        for (int i = 0; i < m1.length; i++) {//row index
            for (int j = 0; j < m2[0].length; j++) {//column index
                for (int k = 0; k < m1[0].length; k++) {
                    result[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return result;
    }

    /**
     * element-wise addition of two matrices of the same shape.
     * @param m1 a 2D array
     * @param m2 another 2D array 
     * @return returns a 2D array with the shape of either m1 or m2.
     */
    public static float[][] mm_add(float[][] m1, float[][] m2) {
        float[][] result = new float[m1.length][m1[0].length];
        for (int i = 0; i < m1.length; i++) {
            for(int j = 0; j < m1[0].length; j++){
            result[i][j] = m1[i][j] + m2[i][j];
        }
        }
        return result;
    }

    /**
     * generates a vector of the specified size and initializes its elements to zero.
     * @param size size of the vector.
     * @return a 2D array of size [1][size]
     */
    public static float[][] v_zeros(int size) {
        float[][] result = new float[1][size];
        for (int i = 0; i < size; i++) {
            result[0][i] = 0.0f;
        }
        return result;
    }

    /**
     * creates a vector of size and assigns the specified value to every element.
     * @param size size of the vector.
     * @param value the value to be assigned to every element.
     * @return a 2D array of size [1][size].
     */
    public static float[][] v_assign(int size, float value) {
        float[][] result = new float[1][size];
        for (int i = 0; i < result[0].length; i++) {
            result[0][i] = value;
        }
        return result;
    }

    /**
     * performs element-wise exponentiation on a vector.
     * @param v the input vector.
     * @return 2D array with the same shape as v.
     */
    public static float[][] v_exp(float[][] v) {
        float[][] exp = new float[1][v[0].length];
        for (int i = 0; i < v[0].length; i++) {
            exp[0][i] = (float) Math.exp(v[0][i]);
        }
        return exp;
    }

    /**
     * performs element-wise scaling on a vector.
     * @param v the input vector to be scaled.
     * @param scale the scaling factor.
     * @return a 2D array with the same shape as v.
     */
    public static float[][] v_scale(float[][] v, float scale) {
        float[][] scl = new float[1][v[0].length];
        for (int i = 0; i < v[0].length; i++) {
            scl[0][i] = (float) v[0][i] * scale;
        }
        return scl;
    }

    /**
     * performs element-wise scaling on a matrix.
     * @param mat the input matrix.
     * @param scale the scaling factor.
     * @return 
     */
    public static float[][] m_scale(float[][] mat, float scale) {
        float[][] scl = new float[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                scl[i][j] = (float) mat[i][j] * scale;
            }
        }
        return scl;
    }

    /**
     * sums all the elements with in a vector.
     * @param v the input vector.
     * @return the total floating sum.
     */
    public static float v_sum(float[][] v) {
        float sum = 0;
        for (int i = 0; i < v[0].length; i++) {
            sum += v[0][i];
        }
        return sum;
    }

    /**
     * returns the transpose of a matrix with a shape of [m][n].
     * @param mat the input matrix.
     * @return a 2D array with a shape of [n][m]. 
     */
    public static float[][] m_transpose(float[][] mat) {
        float[][] transpose = new float[mat[0].length][mat.length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                transpose[j][i] = mat[i][j];
            }
        }
        return transpose;
    }

    /**
     * pretty prints a floating point vector with the specified number of significant figures.
     * @param v the input vector.
     * @param decimal the number of significant figures applied element-wise. 
     */
    public static void v_print(float[] v, int decimal) {
        String pattern = "%7." + decimal + "f";
        for (int j = 0; j < v.length; j++) {
            System.out.printf(pattern, v[j]);
        }
        System.out.println("");
    }

    /**
     * pretty prints a floating point matrix with the specified number of significant figures.
     * @param mat the input matrix.
     * @param decimal the number of significant figures applied element-wise.     
     */
    public static void m_print(float[][] mat, int decimal) {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat.length; j++) {
                System.out.printf("%7.3f", mat[i][j]);
            }
            System.out.println();
        }
    }

    /**
     * returns the square magnitude of a vector.
     * @param v the input vector.
     * @return the floating sum.
     */
    public static float v_sqr_mgn(float[][] v) {
        float sq_sum = 0;
        for (int i = 0; i < v[0].length; i++) {
            sq_sum += v[0][i] * v[0][i];
        }
        return sq_sum;
    }

    /**
     * creates a matrix of size [h] X [w] and initializes it with random values.
     * @param h row size.
     * @param w column size.
     * @return a 2D array of shape [h] X [w]
     */
    public static float[][] m_random(int h, int w) {
        float[][] result = new float[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                result[i][j] = (float) Math.random();
            }
        }
        return result;
    }
    
        /**
     * creates a matrix of size [h] X [w] and initializes it with zeros.
     * @param h row size.
     * @param w column size.
     * @return a 2D array of shape [h] X [w]
     */
    public static float[][] m_zeros(int h, int w) {
        float[][] result = new float[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                result[i][j] = (float) Math.random();
            }
        }
        return result;
    }

    /**
     * creates a row vector of size [w] and initialize it with random values between 0.0 and 1.0.
     * @param w column size
     * @return a 2D array of shape [1] X [w].
     */
    public static float[][] v_random(int w) {
        float[][] result = new float[1][w];
        for (int j = 0; j < w; j++) {
            result[0][j] = (float) Math.random();
        }

        return result;
    }

    /**
     * returns a sub-array from the original matrix with the specified range of indices.
     * @param mat the input matrix.
     * @param r_s start row index.
     * @param r_e end row index.
     * @param c_s start column index.
     * @param c_e end column index.
     * @return 
     */
    public static float[][] m_sub(float[][] mat, int r_s, int r_e, int c_s, int c_e) {
        float[][] sub = new float[r_e - r_s + 1][c_e - c_s + 1];
        for (int i = 0; i < sub.length; i++) {
            for (int j = 0; j < sub[0].length; j++) {
                sub[i][j] = mat[r_s + i][c_s + j];
            }
        }
        return sub;
    }

    /**
     * performs element-wise multiplication between two matrices and sums the result.
     * @param mat1 the first input matrix.
     * @param mat2 the second input matrix.
     * @return the final sum of the product.
     */
    public static float mm_elsum(float[][] mat1, float[][] mat2) {
        float sum = 0;
        for (int i = 0; i < mat1.length; i++) {
            for (int j = 0; j < mat2[0].length; j++) {
                sum += mat1[i][j] * mat2[i][j];
            }
        }
        return sum;
    }

    /**
     * returns the maximum value from a matrix.
     * @param mat the input matrix.
     * @return a floating value 
     */
    public static float m_max(float[][] mat) {
        float max = mat[0][0];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                max = max < mat[i][j] ? mat[i][j] : max;
            }
        }
        return max;
    }

     /**
     * returns the maximum value from a vector.
     * @param vec the input vector.
     * @return a floating value 
     */
    public static float v_max(float[][] vec) {
        float max = vec[0][0];
        for (int i = 0; i < vec[0].length; i++) {
            max = max < vec[0][i] ? vec[0][i] : max;
        }
        return max;
    }

    /**
     * returns the index of the maximum value within a vector.
     * @param vec the input vector.
     * @return an integer that corresponds to the index.
     */
    public static float v_argmax(float[][] vec) {

        int arg = 0;
        for (int i = 0; i < vec[0].length; i++) {
            arg = vec[0][arg] < vec[0][i] ? i : arg;
        }
        return arg;
    }

    // 
    /**
     * pretty prints the shape of a 2D array.
     * @param mat the input matrix.
     */
    public static void m_size(float[][] mat) {
        System.out.println("" + mat.length + " X " + mat[0].length);
    }

       /**
     * flattens a 2D array of shape [m] X [n] into a 2D array of shape
     * [1][m*n]
     * @param mat the input matrix.
     * @return a 2D array of shape [1][m*n*p]
     */
    public static float[][] m_flatten(float[][] mat) {
        float[][] v = new float[1][mat.length * mat[0].length];
        int k = 0; //vector iterator
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                v[0][k] = mat[i][j];
                k++;
            }
        }
        return v;
    }
    
    
    /**
     * flattens a 3D array of shape [m] X [n] X [p] into a 2D array of shape
     * [1][m*n*p]
     * @param mat the input matrix.
     * @return a 2D array of shape [1][m*n*p]
     */
    public static float[][] m_flatten(float[][][] mat) {
        float[][] v = new float[1][mat.length * mat[0].length * mat[0][0].length];
        int l = 0; //vector iterator
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                for (int k = 0; k < mat[0][0].length; k++) {
                    v[0][l] = mat[i][j][k];
                    l++;
                }
            }
        }
        return v;
    }

    /**
     * converts a 1D java array into a row vector of the same size with the extra 
     * dimension. i.e. [n] ----> [1][n].
     * @param input the input 1D vector.
     * @return a 2D array with shape [1]X[n].
     */ 
    public static float[][] m_row(float[] input) {
        float[][] result = new float[1][input.length];
        for (int i = 0; i < input.length; i++) {
            result[0][i] = input[i];
        }
        return result;
    }

     /**
     * converts a 1D java array into a column vector of the same size with the extra 
     * dimension. i.e. [m] ----> [m][1].
     * @param input the input 1D vector.
     * @return a 2D array with shape [m]X[1].
     */ 
    public static float[][] m_column(float[] input) {
        float[][] result = new float[input.length][1];
        for (int i = 0; i < input.length; i++) {
            result[i][0] = input[i];
        }
        return result;
    }
    
    /**
     * reorganizes a matrix into the desired 3D matrix of shape [d][h][w].
     * the size of the input and d*h*w must be the equal.
     * @param input the input matrix.
     * @param d the depth of the reshaped matrix.
     * @param h the row size of the reshaped matrix.
     * @param w the column size of the reshaped matrix.
     * @return 
     */
    public static float[][][] reshape(float[][] input, int d, int h, int w){
        //input --> [1Xn]  output --> [d][h][w]
        float[][][] output=new float[d][h][w];
        int input_index=0;
        for(int i=0;i<d;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    output[i][j][k]=input[0][input_index];
                    input_index++;
                }
            }
        }
        return output;
    }



}
