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
#include <face_mesh.hpp>
#include <face_detection.hpp>
#include <iris_mesh.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

cv::Rect calculate_eye_roi(cv::Point3f &eye_left_side, cv::Point3f &eye_right_side) {
    int center_x = (eye_left_side.x + eye_right_side.x)/2;
    int center_y = (eye_left_side.y + eye_right_side.y)/2;
    int width = std::abs(eye_left_side.x - eye_right_side.x);
    int height = std::abs(eye_left_side.y - eye_right_side.y);
    width = height = std::max(width, height);
    return cv::Rect(center_x - width/2, center_y - height/2, width, height);
}



int main(int argc, char *argv[])
{
    /* Initialize camera */
    const uint8_t camera_index = 0;
    const uint16_t camera_fps = 30;
    const uint32_t width = 1280;
    const uint32_t height = 720;
    cv::VideoCapture cam(camera_index);

    if (cam.isOpened() == false)
    {
        fprintf(stderr, "ERROR: Cannot open camera!\n");
        exit(1);
    }

    cam.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cam.set(cv::CAP_PROP_FRAME_COUNT, camera_fps);

    static CLFML::FaceDetection::FaceDetector face_det;
    static CLFML::FaceMesh::FaceMesh mesh_det;
    static CLFML::IrisMesh::IrisMesh iris_det;

    face_det.load_model(CFML_FACE_DETECTOR_CPU_MODEL_PATH);
    mesh_det.load_model(CFML_FACE_MESH_CPU_MODEL_PATH);
    iris_det.load_model(CLFML_IRIS_MESH_CPU_MODEL_PATH);

    /* Create window to show the face roi */
    cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::resizeWindow("Display window", width, height);

    cv::Mat cam_frame, iris_left_eye, iris_right_eye;
    while (true)
    {
        /* If no frame captured? Try again! */
        if (!cam.read(cam_frame))
        {
            continue;
        }

        /* Load image into model and do inference! */
        face_det.load_image(cam_frame);

        /* Get face_detected value */
        int face_detected = face_det.detected() + 1; // +1 because it returns -1 for no face and 0 for face detected!
                                                     /* Get the face roi rectangle */
        cv::Rect face_roi = face_det.get_face_roi();
        
        if (face_detected)
        {
            if (face_roi.x >= 0 && face_roi.y >= 0 &&
                face_roi.x + face_roi.width <= cam_frame.cols &&
                face_roi.y + face_roi.height <= cam_frame.rows)
            {
                cv::Mat cropped_image_to_roi = cam_frame(face_roi);
                mesh_det.load_image(cropped_image_to_roi, face_roi);

                /* Draw the face roi rectangle on the captured camera frame */
                cv::rectangle(cam_frame, face_roi, cv::Scalar(0, 255, 0), 2); // Green rectangle will be drawn around detected face
                /* Get the face landmarks */
                std::array<cv::Point, 6> face_keypoints = face_det.get_face_landmarks();

                /* Draw the face landmarks on top of the captured camera frame */
                for (cv::Point keypoint : face_keypoints)
                {
                    cv::circle(cam_frame, keypoint, 2, cv::Scalar(0, 255, 0), -1);
                }

                std::array<cv::Point3f, CLFML::FaceMesh::NUM_OF_FACE_MESH_POINTS> face_mesh_keypoints = mesh_det.get_face_mesh_points();

                /* Draw the face landmarks on top of the captured camera frame */
                for (cv::Point3f keypoint : face_mesh_keypoints)
                {
                    cv::circle(cam_frame, cv::Point(keypoint.x, keypoint.y), 2, cv::Scalar(0, 255, 0), -1);
                }
                
                cv::Rect left_eye_roi = calculate_eye_roi(face_mesh_keypoints[446], face_mesh_keypoints[464]);
                cv::rectangle(cam_frame, left_eye_roi, cv::Scalar(255, 0, 0), 2); // Green rectangle will be drawn around detected eye
                iris_left_eye = cam_frame(left_eye_roi);
                iris_det.load_image(iris_left_eye, left_eye_roi);
                std::array<cv::Point3f, CLFML::IrisMesh::NUM_OF_IRIS_MESH_POINTS> iris_mesh_keypoints = iris_det.get_iris_mesh_points();

                for (cv::Point3f keypoint : iris_mesh_keypoints)
                {
                    cv::circle(cam_frame, cv::Point(keypoint.x, keypoint.y), 2, cv::Scalar(0, 255, 0), -1);
                }

                cv::Rect right_eye_roi = calculate_eye_roi(face_mesh_keypoints[244], face_mesh_keypoints[226]);
                cv::rectangle(cam_frame, right_eye_roi, cv::Scalar(255, 0, 0), 2); // Green rectangle will be drawn around detected eye
                iris_right_eye = cam_frame(right_eye_roi);
                iris_det.load_image(iris_right_eye, right_eye_roi);
                iris_mesh_keypoints = iris_det.get_iris_mesh_points();
                for (cv::Point3f keypoint : iris_mesh_keypoints)
                {
                    cv::circle(cam_frame, cv::Point(keypoint.x, keypoint.y), 2, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        /* Convert the face_detected integer to string */
        const std::string top_left_text = "Detected: " + std::to_string(face_detected);
        /* Draw (red) text in corner of frame telling whether a face has been detected; 0 no face, 1 face has been detected */
        cv::putText(cam_frame, top_left_text, cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2);
        /* Update the window with the newly made image */
        cv::imshow("Display window", cam_frame);

        /* Break the loop on 'q' key press */
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    return 0;
}
