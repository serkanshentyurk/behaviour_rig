using Bonsai;
using Bonsai.Dsp;
using MathNet.Numerics;
using OpenCV.Net;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Reactive;
using Bonsai.Expressions;

[Combinator]
[Description("Implements an antibias with buffered values and a gamma kernel")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Antibias
{
    
    public Antibias()
    {
        alpha = 2.0F;
        beta  = 0.2F;
        slope = 10F;
    }

    /// <summary>
    /// Gets or sets the alpha (shape) parameter of the gamma kernel 
    /// </summary>
    [Description("The alpha parameter of the gamma kernel")]
    public float alpha { get; set; }

    /// <summary>
    /// Gets or sets the beta (rate) parameter of the gamma kernel
    /// </summary>
    [Description("The beta parameter of the beta kernel")]
    public float beta { get; set; }

    /// <summary>
    /// Gets or sets the slope parameter of the sigmoid transfer function 
    /// </summary>
    [Description("The slope parameter of the sigmoid transfer function")]
    public float slope { get; set; }

    public IObservable<Tuple<float,float,float>> Process(IObservable<Mat> source)
    {
        return source.Select(input =>
        {
            float[] kernel = gamma_kernel(alpha, beta, input.Cols);
            float weightedSum = 0F;
            float standardSum = 0F;
            float choice;
            // get the data
            float[] dataArray = new float[input.Cols];
            for (int i = 0; i < input.Cols; i++)
            {
                dataArray[i] = (float)input[i].Val0;
            }
            Array.Reverse(dataArray);
            for (int i = 0; i < input.Cols; i++)
            {
                choice = dataArray[i];
                weightedSum += kernel[i] * choice;
                standardSum += choice * 1 / input.Cols;
            }           
            float slopeParam = slope;
            float prob = sigmoid(weightedSum, slopeParam);
            return Tuple.Create(prob,standardSum,weightedSum);
        });
    }

    public static float[] gamma_kernel(float alpha, float beta, int count)
    {
        var gamma_alpha = SpecialFunctions.Gamma(alpha);
        float[] kernel = new float[count];
        float norm = 0;
        float value;
        for (int i = 0; i < count; i++)
        {
            value = (float)i + 1F / beta;
            kernel[i] = (float)((Math.Pow(value, alpha - 1) * Math.Exp(-beta * value) * Math.Pow(beta,alpha))/gamma_alpha);
            norm += kernel[i];
        }
        for (int i = 0; i < count; i++)
        {
            kernel[i] /= norm;
        }
        return kernel;
    }

    public static float sigmoid(float argument, float slopeParam)
    {
        float prob = 1 / (1 + (float)Math.Exp(-slopeParam * (argument - 0.5)));
        return prob;
    }


}
