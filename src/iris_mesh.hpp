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
#ifndef IRIS_MESH_HPP
#define IRIS_MESH_HPP
#include <string>
#include <cstdint>
#include <opencv2/core.hpp>
#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/model.h>

namespace CLFML::IrisMesh
{
    /* Number of iris landmarks */
    inline constexpr size_t NUM_OF_IRIS_MESH_POINTS = 5;

    /* Number of eye contours and brows points */
    inline constexpr size_t NUM_OF_IRIS_MESH_CONT_BROWS_POINTS = 71; 

    /* Number of model output tensors */
    inline constexpr size_t NUM_OF_FACE_MESH_OUTPUT_TENSORS = 2;

    /*
     * The used delegate for model inference;
     * In the future TPU support might be added
     */
    enum class face_mesh_delegate
    {
        CPU
    };

    class IrisMesh
    {
    public:
        IrisMesh();

        /**
         * @brief Loads model and initializes the inference runtime
         * @param model_path Path to the Mediapipe Face Mesh model (.tflite) file
         * @param delegate_type The delegate to use for inference (CPU only for now) (default = CPU)
         * @param num_of_threads The number of CPU threads which can be used by the inference runtime (default= 4 threads)
         */
        void load_model(const std::string model_path, const face_mesh_delegate delegate_type = face_mesh_delegate::CPU, const uint8_t num_of_threads = 4);

        /**
         * @brief Loads image into model and does inference
         * @param frame Any frame which is formatted in CV_8UC3 or CV_8UC4 format
         * @param roi_offset If the input_frame is a cropped ROI frame, 
         *                   the face_mesh_points can be adjusted to the original frame.
         */
        void load_image(cv::Mat &frame, cv::Rect roi_offset = cv::Rect(0, 0, 0, 0));

        /**
         * @brief Get the 3D landmarks from the model
         * @return Array with 5x3D Iris landmarks;
         */
        std::array<cv::Point3f, NUM_OF_IRIS_MESH_POINTS> get_iris_mesh_points();

        std::array<cv::Point3f, NUM_OF_IRIS_MESH_CONT_BROWS_POINTS> get_iris_brow_points();

        ~IrisMesh();

    private:
        /* Model input frame width and height (set to defaults from the model-card) 
         * See: https://drive.google.com/file/d/1bsWbokp9AklH2ANjCfmjqEzzxO1CNbMu/preview
         */
        int32_t m_input_frame_size_x = 64;

        int32_t m_input_frame_size_y = 64;

        int32_t m_proc_frame_size_x;

        int32_t m_proc_frame_size_y;

        cv::Rect m_roi_offset;

        /*
         * Model inputs and outputs
         */
        TfLiteTensor *m_input_tensor;

        /*
         * The output tensors of the model, there are two outputs; 
         * - The 5x3D Landmark points (3x32-bit float)
         * - The Eye contours and brows (71x3D 32-bit float)
         */
        std::array<TfLiteTensor *, NUM_OF_FACE_MESH_OUTPUT_TENSORS> m_output_tensors;

        /* Array that contains the unprocessed Iris Mesh landmarks */
        std::array<float, NUM_OF_IRIS_MESH_POINTS * 3> m_model_mesh_point_output;

        /* Array that contains the unprocessed Iris Mesh Brows and Contours points */
        std::array<float, NUM_OF_IRIS_MESH_CONT_BROWS_POINTS * 3> m_model_brows_contours_output;

        /* Internal variable that contains the 3D Landmarks */
        std::array<cv::Point3f, NUM_OF_IRIS_MESH_POINTS> m_model_landmarks;

        std::array<cv::Point3f, NUM_OF_IRIS_MESH_CONT_BROWS_POINTS> m_model_brow_points;
        /*
         * Handles to the model and model_inpreter runtime
         */
        std::unique_ptr<tflite::FlatBufferModel> m_model;
        std::unique_ptr<tflite::Interpreter> m_model_interpreter;

        /**
         * @brief Preprocess any incoming image to a 64x64px 24-bit RGB image
         */
        cv::Mat preprocess_image(const cv::Mat &in);

        /**
         * @brief Copies the results from the tensors into our m_model_regressors array
         */
        void get_regressor();
    };
}

#endif /* IRIS_MESH_HPP */