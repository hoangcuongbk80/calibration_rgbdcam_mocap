#include <ros/ros.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <pcl/visualization/pcl_visualizer.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>

#include <Eigen/Geometry>
#include <pcl/common/transforms.h>

#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

bool run_getChessboardPose_inCamCS = false;
std::string rgb_dir, depth_dir;
cv::Mat rgb_img, depth_img;
std::string saved_draw_dir;
float cx, cy, fx, fy;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;
Eigen::Vector3f corner_pos[3];

bool run_getT_camToMarkers = false;
std::string results_dir;

void depthToPoints()
{
  depth_img = imread (depth_dir, CV_LOAD_IMAGE_UNCHANGED);
  std::cerr << "Load depth image from " << depth_dir << "\n";
  cloud_ .reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB point;
  for(int row = 0; row < rgb_img.rows; row++)
  {
    for(int col = 0; col < rgb_img.cols; col++)       
    {
      if(isnan(depth_img.at<unsigned short>(row, col))) continue;
      unsigned short depth = depth_img.at<unsigned short>(row, col);
      point.x = (col-cx) * depth / fx;
      point.y = (row-cy) * depth / fy;
      point.z = depth;

      Vec3b intensity = rgb_img.at<Vec3b>(row, col);
      uchar blue = intensity.val[0];
      uchar green = intensity.val[1];
      uchar red = intensity.val[2];
      point.r = blue; 
      point.b = green; 
      point.g = red;
      cloud_->push_back(point);
    }
  }
}

pcl::PointXYZ get3dpoint(int const col, int const row)
{
  float corDepth = 0;

  int count = 0;
  int radius = 5;
  for(int i=-radius; i <=radius ; i++ )
  for(int j=-radius; j <=radius ; j++)
    if (!isnan(depth_img.at<unsigned short>(row, col)))
    if(row < depth_img.rows & row > -1)
    if(col < depth_img.cols & col > -1)      
    {
      count++;
      corDepth += depth_img.at<unsigned short>(row+i, col+j);  
    }
  if(!count)
  {
    std::cerr << "Fail! depth is nan at corner of chessboard." << "\n";
    exit(EXIT_FAILURE);
  }
  corDepth = (float) corDepth / count;  
  pcl::PointXYZ point;
  point.x = (col-cx) * corDepth / fx;
  point.y = (row-cy) * corDepth / fy;
  point.z = corDepth;
  return point;
}

void find_chessboard_corners()
{
  rgb_img = imread (rgb_dir, -1);
  std::cerr << "Load rgb image from " << rgb_dir << "\n";

  Size imageSize = rgb_img.size();
  int boardsize_X = 6;
  int boardsize_Y = 10;
  Size boardSize(boardsize_Y, boardsize_X); float squareSize = 0.065;
  std::vector<std::vector<cv::Point2f> > imagePoints(1); //corners

  bool found = false;
  int chessBoardFlags =  CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE;
  found = findChessboardCorners( rgb_img, boardSize, imagePoints[0], chessBoardFlags);
  if (found)
  {
    drawChessboardCorners( rgb_img, boardSize, Mat(imagePoints[0]), found );
    putText(rgb_img, "4", imagePoints[0][0], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    putText(rgb_img, "1", imagePoints[0][9], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    putText(rgb_img, "2", imagePoints[0][50], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    putText(rgb_img, "3", imagePoints[0][59], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

    std::cerr << imagePoints[0][0].x << " " << imagePoints[0][0].y << "\n";
    std::cerr << "Save draw chessboard to " << saved_draw_dir << "\n";
    cv::imwrite(saved_draw_dir, rgb_img);
  }
  else
  {
    std::cerr << "Coudn't find Chessboard Corners" << "\n";
    return;
  }
  depthToPoints();
  
  pcl::PointXYZ point;
  int col, row;
  //corner 4
  col = (int)imagePoints[0][0].x;
  row = (int)imagePoints[0][0].y;
  point = get3dpoint(col, row);
  corner_pos[0] << point.x,  point.y, point.z;
  //corner 1
  col = (int)imagePoints[0][9].x;
  row = (int)imagePoints[0][9].y;
  point = get3dpoint(col, row);
  corner_pos[1] << point.x,  point.y, point.z;
  //corner 2
  col = (int)imagePoints[0][50].x;
  row = (int)imagePoints[0][50].y;
  point = get3dpoint(col, row);
  corner_pos[2] << point.x, point.y, point.z;
  //corner 3
  col = (int)imagePoints[0][59].x;
  row = (int)imagePoints[0][59].y;
  point = get3dpoint(col, row);
  corner_pos[3] << point.x,  point.y, point.z;
}

void chessboardPose__inCameraCS()
{
  // Find chessboard pose with respect to camera coordinate system

  find_chessboard_corners();
  Eigen::Affine3f BoardPose_CamCS = Eigen::Affine3f::Identity();
  Eigen::Vector3f corners_center, cor13_center, cor23_center;
  corners_center = (corner_pos[0] + corner_pos[1] + corner_pos[2] + corner_pos[3]) / 4;
  cor13_center = (corner_pos[1] + corner_pos[3]) / 2; //check markers coordinate system on chessboard from mocap
  cor23_center = (corner_pos[2] + corner_pos[3]) / 2;
  Eigen::Vector3f x1,y1,z1;  // unit vectors
  x1 = (cor23_center - corners_center).normalized();
  y1 = (cor13_center - corners_center).normalized();
  z1 = x1.cross(y1);

  pcl::getTransformationFromTwoUnitVectorsAndOrigin(y1, z1, corners_center, BoardPose_CamCS);
  pcl::transformPointCloud (*cloud_, *cloud_, BoardPose_CamCS);
  std::cerr << "\nChessboard position with respect to camera coordinate system: x y z \n" << 
  BoardPose_CamCS.translation()(0,0) << " " << BoardPose_CamCS.translation()(1,0) << " " << BoardPose_CamCS.translation()(2, 0) << "\n";

  Eigen::Quaternionf q(BoardPose_CamCS.linear());
  std::cerr << "Chessboard Quaternion with respect to camera coordinate system: rx ry rz w \n" << 
  q.vec()(0,0) << " " << q.vec()(1,0) << " " << q.vec()(2,0) << " " << q.w() << "\n";
  std::cerr << "BoardPose_CamCS: \n" << BoardPose_CamCS.matrix() << "\n";
}

std::vector<Eigen::Affine3f> readFile(std::string name)
{
  std::vector<Eigen::Affine3f> chessboard_Pose_inCamCS;
  Eigen::Affine3f Pos = Eigen::Affine3f::Identity();

  std::string fileName = results_dir;
  fileName.append(name);
  std::cerr << "fileName: " << fileName << "\n";

  std::ifstream infile(fileName);
  std::string line;
  while (std::getline(infile, line))
  {
    std::istringstream iss(line);
    Eigen::Quaternionf q;
    if (!(iss >> Pos.translation()(0,0) >> Pos.translation()(1,0) >> Pos.translation()(2,0) 
              >> q.vec()(0,0) >> q.vec()(1,0) >> q.vec()(2,0) >> q.w())) 
    {
      std::cerr << "Couldn't read a line in " << fileName << "\n";
      exit(1);
    }
    Pos.linear() = q.normalized().toRotationMatrix();
    std::cerr << "Pos: \n" << Pos.matrix() << "\n";
    chessboard_Pose_inCamCS.push_back(Pos);
  }
  return chessboard_Pose_inCamCS;
}

void getT_camToMarkers()
{
  std::vector<Eigen::Affine3f> chessboard_Pose_inCamCS;
  chessboard_Pose_inCamCS = readFile("PoseOfMarkersOnCam_inCamCS.txt");
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "find_transform");
  ros::NodeHandle nh_;  
  
  nh_ = ros::NodeHandle("~");
  nh_.getParam( "run_getChessboardPose_inCamCS", run_getChessboardPose_inCamCS );  
  nh_.getParam( "rgb_dir", rgb_dir );
  nh_.getParam( "depth_dir", depth_dir );
  nh_.getParam( "saved_draw_dir", saved_draw_dir );
  nh_.getParam( "cx", cx );
  nh_.getParam( "cy", cy );
  nh_.getParam( "fx", fx );
  nh_.getParam( "fy", fy );

  nh_.getParam( "run_getT_camToMarkers", run_getT_camToMarkers );
  nh_.getParam( "results_dir", results_dir );

  if(run_getChessboardPose_inCamCS) 
  {
    std::cerr << "\n--------Run chessboardPose__inCameraCS()---------\n";
    chessboardPose__inCameraCS();
    std::cerr << "--------------------------000-------------------------------\n";

  }
  if(run_getT_camToMarkers)
  {
    std::cerr << "\n----------------Run getT_camToMarkers()-------------------\n";
    getT_camToMarkers();
    std::cerr << "--------------------------000-------------------------------\n";
  }

  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0.7, 0.7, 0.72);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud_, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
  viewer->addCoordinateSystem (500.0);
  viewer->initCameraParameters ();
  while (!viewer->wasStopped ())
    {
        viewer->spinOnce (1000);
    }

  ros::spin();
  return 0;
}