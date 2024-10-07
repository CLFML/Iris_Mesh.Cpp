/*
*  Copyright 2024 (C) Jeroen Veen <ducroq> & Victor Hogeweij <Hoog-V>
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*  http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
* This file is part of the Iris_Mesh.Cpp library
*
* Author:          Victor Hogeweij <Hoog-V>
*
*/

#include <iris_mesh.hpp>
#include "tensorflow/lite/kernels/register.h"
#include "opencv2/imgproc.hpp"

namespace CLFML::IrisMesh
{
    /* Mapping from model output tensor name to their array index */
    enum output_tensor_id
    {
        OUTPUT_TENSOR_BROWS_CONTOUR = 0,
        OUTPUT_TENSOR_LANDMARKS = 1
    };

    IrisMesh::IrisMesh()
    {
    }

    void IrisMesh::load_model(const std::string model_path, const face_mesh_delegate delegate_type, const uint8_t num_of_threads)
    {

        /* Load the model in to memory */
        m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (m_model == nullptr)
        {
            fprintf(stderr, "File \"%s\" ERROR: Can't build flatbuffer from model: %s \n", __FILE__, model_path.c_str());
            exit(1);
        }

        /*
         * We want to use the default tensorflow operations (for CPU inference):
         * See: https://www.tensorflow.org/lite/guide/ops_compatibility
         */
        static tflite::ops::builtin::BuiltinOpResolver resolver;

        /*
         * Build the model interpreter with our model and the resolver;
         * This gives a handle and saves it into m_model_intepreter which is later used for doing inference
         */
        if (tflite::InterpreterBuilder(*m_model, resolver)(&m_model_interpreter) != kTfLiteOk)
        {
            fprintf(stderr, "File \"%s\" ERROR: Can't initialize the interpreter \n", __FILE__);
            exit(1);
        }

        /*
         * We can set the amount of CPU threads we want to dedicate to our model_interpreter engine.
         * This is only useful for CPU inference,
         * Using multiple threads with TPU inference will slow down the program, due to synchronisation between threads!
         * Default = 4 threads
         */
        m_model_interpreter->SetNumThreads(num_of_threads);

        /* Allocate memory for model inference */
        if (m_model_interpreter->AllocateTensors() != kTfLiteOk)
        {
            fprintf(stderr, "File \"%s\" ERROR: Can't allocate tensors for face detector model interpreter \n", __FILE__);
            exit(1);
        }

        /*
         * Get the amount of input tensors (vectors) of the model, which should be one
         * As this is where the 128x128 input frame will be copied to before doing inference
         */
        static const std::vector<int> &inputs = m_model_interpreter->inputs();

        /* Save the handle to this input vector for later */
        m_input_tensor = m_model_interpreter->tensor(inputs.at(0));

        /* Get the input frame size (should be 64x64 pixels!) */
        m_input_frame_size_x = m_input_tensor->dims->data[1];
        m_input_frame_size_y = m_input_tensor->dims->data[2];

        /*
         * Get the amount of output tensors of the model, which should be two;
         *  - one for the list of Eye contours
         *  - one for the list of Iris keypoints
         */
        const std::vector<int> &outputs = m_model_interpreter->outputs();

        if (outputs.size() != m_output_tensors.size())
        {
            fprintf(stderr, "File \"%s\" ERROR: Model tensor quantity does not match expected tensors! \n", __FILE__);
            exit(1);
        }

        /* Save the output tensor handles for later! (With bound-check)*/
        for (uint8_t i = 0; i < m_output_tensors.size(); i++)
        {
            int tensor_index = outputs.at(i);
            m_output_tensors.at(i) = m_model_interpreter->tensor(tensor_index);
        }
    }

    void IrisMesh::get_regressor()
    {
        size_t num_of_bytes = m_output_tensors.at(OUTPUT_TENSOR_LANDMARKS)->bytes;
        size_t num_of_floats = num_of_bytes / sizeof(float);

        num_of_floats = num_of_bytes / sizeof(float);
        memcpy(&(m_model_mesh_point_output.at(0)), m_output_tensors.at(OUTPUT_TENSOR_LANDMARKS)->data.f, num_of_bytes);

        num_of_bytes = m_output_tensors.at(OUTPUT_TENSOR_BROWS_CONTOUR)->bytes;
        num_of_floats = num_of_bytes / sizeof(float);
        memcpy(&(m_model_brows_contours_output.at(0)), m_output_tensors.at(OUTPUT_TENSOR_BROWS_CONTOUR)->data.f, num_of_bytes);
    }

    /**q
     *
     * @brief This function converts images with other color-spaces to RGB.
     *        As the model expects RGB formatted images.
     * @param in The image to convert to RGB,
     *           Can be CV_8UC3; 8-bit int with 3 channels
     *           Or CV_8UC4; 8-bit int with 4 channels
     * @return RGB formatted frame
     *
     */
    cv::Mat convert_image_to_rgb(const cv::Mat &in)
    {
        cv::Mat rgb_frame;
        int frame_color_type = in.type();

        switch (frame_color_type)
        {
        case CV_8UC3:
        {
            cv::cvtColor(in, rgb_frame, cv::COLOR_BGR2RGB);
            break;
        }
        case CV_8UC4:
        {
            cv::cvtColor(in, rgb_frame, cv::COLOR_BGRA2RGB);
            break;
        }
        default:
        {
            fprintf(stderr, "ERROR: Image type %d is not supported by the face_detector library! \n", frame_color_type);
            exit(1);
        }
        };
        return rgb_frame;
    }

    cv::Mat IrisMesh::preprocess_image(const cv::Mat &in)
    {
        cv::Mat preprocessed_frame = convert_image_to_rgb(in);
        cv::Size input_frame_size = cv::Size(m_input_frame_size_x, m_input_frame_size_y);
        cv::resize(preprocessed_frame, preprocessed_frame, input_frame_size);

        const double alpha = 1 / 127.5f;
        const double beta = -127.5f / 127.5f;
        preprocessed_frame.convertTo(preprocessed_frame, CV_32FC3, alpha, beta);
        return preprocessed_frame;
    }

    void IrisMesh::load_image(cv::Mat &frame, cv::Rect roi_offset)
    {
        m_roi_offset = roi_offset;
        m_proc_frame_size_x = frame.size().width;
        m_proc_frame_size_y = frame.size().height;

        /* Convert image to 64x64 pixels image with CV32_FC3 format */
        cv::Mat preprocessed_image = preprocess_image(frame);

        /* Copy the image data to the input tensor, which feeds it into the model */
        memcpy(m_input_tensor->data.f, preprocessed_image.data, m_input_tensor->bytes);

        /* Run inference! */
        m_model_interpreter->Invoke();

        /* Get model landmarks */
        get_regressor();


    }

    std::array<cv::Point3f, NUM_OF_IRIS_MESH_POINTS> IrisMesh::get_iris_mesh_points()
    {
        /* Scale the model landmarks to original image dimensions */
        double _x;
        double _y;
        for (size_t point_idx = 0; point_idx < NUM_OF_IRIS_MESH_POINTS; point_idx++)
        {
            cv::Point3f *point = &(m_model_landmarks.at(point_idx));

            /* Do some linear scaling :) */
            _x = (m_model_mesh_point_output.at(point_idx * 3)) / m_input_frame_size_x;
            _y = (m_model_mesh_point_output.at((point_idx * 3) + 1)) / m_input_frame_size_y;
            point->z  = m_model_mesh_point_output.at((point_idx * 3) + 2);

            /* Add original ROI offset if needed */
            point->x = (_x*m_proc_frame_size_x) + m_roi_offset.x;
            point->y = (_y*m_proc_frame_size_y) + m_roi_offset.y;
        }
        return m_model_landmarks;
    }

    std::array<cv::Point3f, NUM_OF_IRIS_MESH_CONT_BROWS_POINTS> IrisMesh::get_iris_brow_points() {
        /* Scale the model landmarks to original image dimensions */
        double _x;
        double _y;
        for (size_t point_idx = 0; point_idx < NUM_OF_IRIS_MESH_POINTS; point_idx++)
        {
            cv::Point3f *point = &(m_model_brow_points.at(point_idx));

            /* Do some linear scaling :) */
            _x = (m_model_brows_contours_output.at(point_idx * 3)) / m_input_frame_size_x;
            _y = (m_model_brows_contours_output.at((point_idx * 3) + 1)) / m_input_frame_size_y;
            point->z  = m_model_brows_contours_output.at((point_idx * 3) + 2);

            /* Add original ROI offset if needed */
            point->x = (_x*m_proc_frame_size_x) + m_roi_offset.x;
            point->y = (_y*m_proc_frame_size_y) + m_roi_offset.y;
        }
        return m_model_brow_points;
    }

    IrisMesh::~IrisMesh()
    {
    }
}
