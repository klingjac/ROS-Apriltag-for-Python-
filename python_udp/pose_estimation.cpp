#include <iostream>
#include <opencv2/opencv.hpp>
#include <apriltag.h>
#include <tagStandard41h12.h>
#include <apriltag_pose.h>

extern "C" {
    // Define the Pose and Transform Structs for the specific detections

    struct Pose {
        double tvec[3];
        double rod_vect[3];
    };

    struct Detection {
        int id;
        Pose pose;
        bool valid = false;
    };

    struct Detections {
        Detection detection_vect[6];
    };

    // Helper function to convert a rotation matrix to Euler angles
    cv::Vec3d rotationMatrixToEulerAngles(const cv::Matx33d &R) {
        double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

        bool singular = sy < 1e-6;

        double x, y, z;
        if (!singular) {
            x = atan2(R(2, 1), R(2, 2));
            y = atan2(-R(2, 0), sy);
            z = atan2(R(1, 0), R(0, 0));
        } else {
            x = atan2(-R(1, 2), R(1, 1));
            y = atan2(-R(2, 0), sy);
            z = 0;
        }
        return cv::Vec3d(x, y, z);
    }

    class AprilTagDetector {
    public:
        AprilTagDetector() {
            tf = tagStandard41h12_create();
            td = apriltag_detector_create();
            apriltag_detector_add_family(td, tf);
            td->nthreads = 4;
            td->quad_decimate = 1.0;
        }

        ~AprilTagDetector() {
            apriltag_detector_destroy(td);
            tagStandard41h12_destroy(tf);
        }

        Detections estimate_pose_and_draw(unsigned char* image_data, int width, int height, double fx, double fy, double cx, double cy, double tag_size) {
            cv::Mat frame(height, width, CV_8UC3, image_data);
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            Detections iteration_detections;

            image_u8_t im = {.width = gray.cols, .height = gray.rows, .stride = gray.cols, .buf = gray.data};
            zarray_t* detections = apriltag_detector_detect(td, &im);

            //std::cout << "Detections: " << zarray_size(detections) << "\n";

            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t* det;
                zarray_get(detections, i, &det);

                // AprilTag pose estimation
                apriltag_detection_info_t info;
                info.det = det;
                info.tagsize = tag_size;
                info.fx = fx;
                info.fy = fy;
                info.cx = cx;
                info.cy = cy;

                apriltag_pose_t pose;
                double err = estimate_tag_pose(&info, &pose);

                // Convert the rotation matrix to a Rodrigues vector
                cv::Mat R(3, 3, CV_64F);
                R.at<double>(0, 0) = MATD_EL(pose.R, 0, 0);
                R.at<double>(0, 1) = MATD_EL(pose.R, 0, 1);
                R.at<double>(0, 2) = MATD_EL(pose.R, 0, 2);
                R.at<double>(1, 0) = MATD_EL(pose.R, 1, 0);
                R.at<double>(1, 1) = MATD_EL(pose.R, 1, 1);
                R.at<double>(1, 2) = MATD_EL(pose.R, 1, 2);
                R.at<double>(2, 0) = MATD_EL(pose.R, 2, 0);
                R.at<double>(2, 1) = MATD_EL(pose.R, 2, 1);
                R.at<double>(2, 2) = MATD_EL(pose.R, 2, 2);
                cv::Mat rvec;
                cv::Rodrigues(R, rvec);
                cv::Matx33d rotation_matrix;
                cv::Rodrigues(rvec, rotation_matrix);
                cv::Vec3d euler_angles = rotationMatrixToEulerAngles(rotation_matrix);

                // Translation vector
                cv::Mat tvec(3, 1, CV_64F);
                tvec.at<double>(0) = MATD_EL(pose.t, 0, 0);
                tvec.at<double>(1) = MATD_EL(pose.t, 1, 0);
                tvec.at<double>(2) = MATD_EL(pose.t, 2, 0);

                // Define the 3D coordinates of the box vertices
                std::vector<cv::Point3f> box_points = {
                    {static_cast<float>(-tag_size / 2), static_cast<float>(-tag_size / 2), 0},
                    {static_cast<float>(tag_size / 2), static_cast<float>(-tag_size / 2), 0},
                    {static_cast<float>(tag_size / 2), static_cast<float>(tag_size / 2), 0},
                    {static_cast<float>(-tag_size / 2), static_cast<float>(tag_size / 2), 0},
                    {static_cast<float>(-tag_size / 2), static_cast<float>(-tag_size / 2), -static_cast<float>(tag_size)},
                    {static_cast<float>(tag_size / 2), static_cast<float>(-tag_size / 2), -static_cast<float>(tag_size)},
                    {static_cast<float>(tag_size / 2), static_cast<float>(tag_size / 2), -static_cast<float>(tag_size)},
                    {static_cast<float>(-tag_size / 2), static_cast<float>(tag_size / 2), -static_cast<float>(tag_size)}
                };

                // Camera intrinsic parameters
                cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
                cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

                // Project the 3D box points to 2D image points
                std::vector<cv::Point2f> img_points;
                cv::projectPoints(box_points, rvec, tvec, camera_matrix, dist_coeffs, img_points);

                // Draw the box
                for (int j = 0; j < 4; j++) {
                    cv::line(frame, img_points[j], img_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
                    cv::line(frame, img_points[j + 4], img_points[(j + 1) % 4 + 4], cv::Scalar(0, 255, 0), 2);
                    cv::line(frame, img_points[j], img_points[j + 4], cv::Scalar(0, 255, 0), 2);
                }

                // Define the coordinate axes for visualization
                std::vector<cv::Point3f> axis_points = {
                    {0, 0, 0},
                    {static_cast<float>(tag_size), 0, 0},
                    {0, static_cast<float>(tag_size), 0},
                    {0, 0, -static_cast<float>(tag_size)}
                };

                // Project the coordinate axes to 2D image points
                std::vector<cv::Point2f> img_axis_points;
                cv::projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, img_axis_points);

                // Draw the coordinate axes
                cv::line(frame, img_axis_points[0], img_axis_points[1], cv::Scalar(0, 0, 255), 2); // X-axis in red
                cv::line(frame, img_axis_points[0], img_axis_points[2], cv::Scalar(0, 255, 0), 2); // Y-axis in green
                cv::line(frame, img_axis_points[0], img_axis_points[3], cv::Scalar(255, 0, 0), 2); // Z-axis in blue

                // Create Pose struct
                Pose pose_curr;
                pose_curr.tvec[0] = tvec.at<double>(0);
                pose_curr.tvec[1] = tvec.at<double>(1);
                pose_curr.tvec[2] = tvec.at<double>(2);

                pose_curr.rod_vect[0] = rvec.at<double>(0);
                pose_curr.rod_vect[1] = rvec.at<double>(1);
                pose_curr.rod_vect[2] = rvec.at<double>(2);

                // Create Detection struct
                Detection detc_curr;
                detc_curr.valid = true;
                detc_curr.pose = pose_curr;
                detc_curr.id = det->id;

                //std::cout << detc_curr.id << "\n";

                // Store the detection
                if (det->id < 6) {
                    iteration_detections.detection_vect[det->id] = detc_curr;
                }
                //std::cout << detc_curr.id << "\n";
                // Draw the pose information for tag with id 2
                if (det->id == 2) {
                    std::ostringstream pose_info;
                    pose_info << "Tvec: [" << tvec.at<double>(0) << ", " << tvec.at<double>(1) << ", " << tvec.at<double>(2) << "]";
                    cv::putText(frame, pose_info.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

                    pose_info.str("");
                    pose_info.clear();
                    pose_info << "Euler Angles: [" << euler_angles[0] * 180 / CV_PI << ", " << euler_angles[1] * 180 / CV_PI << ", " << euler_angles[2] * 180 / CV_PI << "]";
                    cv::putText(frame, pose_info.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                }
                //std::cout << "I'm cooked" << "\n";
            }

            apriltag_detections_destroy(detections);
            //std::cout << "Before the return" << "\n";
            return iteration_detections;
        }

    private:
        apriltag_family_t* tf;
        apriltag_detector_t* td;
    };

    static AprilTagDetector detector;

    Detections estimate_pose_and_draw(unsigned char* image_data, int width, int height, double fx, double fy, double cx, double cy, double tag_size) {
        //std::cout << "fml" << "\n";
        return detector.estimate_pose_and_draw(image_data, width, height, fx, fy, cx, cy, tag_size);
    }
}
