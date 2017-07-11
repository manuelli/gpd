#include "../../../gpd/include/nodes/grasp_detection_node.h"


/** constants for input point cloud types */
const int GraspDetectionNode::POINT_CLOUD_2 = 0; ///< sensor_msgs/PointCloud2
const int GraspDetectionNode::CLOUD_INDEXED = 1; ///< cloud with indices
const int GraspDetectionNode::CLOUD_SAMPLES = 2; ///< cloud with (x,y,z) samples
const int GraspDetectionNode::CLOUD_GRASPS = 3; ///< cloud with (x,y,z) samples



GraspDetectionNode::GraspDetectionNode(ros::NodeHandle& node) : has_cloud_(false), has_normals_(false),
  size_left_cloud_(0), has_samples_(true), frame_("")
{
  cloud_camera_ = NULL;

  // set camera viewpoint to default origin
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  view_point_ << camera_position[0], camera_position[1], camera_position[2];

  // choose sampling method for grasp detection
  node.param("use_importance_sampling", use_importance_sampling_, false);

  if (use_importance_sampling_)
  {
    importance_sampling_ = new SequentialImportanceSampling(node);
  }
  grasp_detector_ = new GraspDetector(node);

  // Read input cloud and sample ROS topics parameters.
  node.param("cloud_type", cloud_type_, POINT_CLOUD_2);
  std::string cloud_topic;
  node.param("cloud_topic", cloud_topic, std::string("/camera/depth_registered/points"));
  std::string samples_topic;
  node.param("samples_topic", samples_topic, std::string(""));
  std::string rviz_topic;
  node.param("rviz_topic", rviz_topic, std::string(""));

  //alternate grasp decoder if needed
  node.param("use_alternate_grasp_msg_decoder", use_alternate_grasp_msg_decoder_, false);

  if (!rviz_topic.empty())
  {
    grasps_rviz_pub_ = node.advertise<visualization_msgs::MarkerArray>(rviz_topic, 1);
    use_rviz_ = true;
  }
  else
  {
    use_rviz_ = false;
  }

  // subscribe to input point cloud ROS topic
  if (cloud_type_ == POINT_CLOUD_2)
    cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_callback, this);
  else if (cloud_type_ == CLOUD_INDEXED)
    cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_indexed_callback, this);
  else if (cloud_type_ == CLOUD_SAMPLES)
  {
    cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_samples_callback, this);
    //    grasp_detector_->setUseIncomingSamples(true);
    has_samples_ = false;
  }
  else if (cloud_type_ == CLOUD_GRASPS)
  {
    std::cout << "using CLOUD_GRASPS callback" << std::endl;
    std::cout << "cloud_topic = " << cloud_topic << std::endl;
    cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_grasps_callback, this);
    //    grasp_detector_->setUseIncomingSamples(true);
    has_samples_ = false;
  }

  // subscribe to input samples ROS topic
  if (!samples_topic.empty())
  {
    samples_sub_ = node.subscribe(samples_topic, 1, &GraspDetectionNode::samples_callback, this);
    has_samples_ = false;
  }

  // uses ROS topics to publish grasp candidates, antipodal grasps, and grasps after clustering
  if (cloud_type_ == CLOUD_GRASPS){
    grasps_pub_ = node.advertise<gpd::GraspConfigList>("clustered_grasps_lucas", 10);
  } else{
    grasps_pub_ = node.advertise<gpd::GraspConfigList>("clustered_grasps", 10);
  }


  node.getParam("workspace", workspace_);
}


void GraspDetectionNode::run()
{
  ros::Rate rate(100);
  ROS_INFO("Waiting for point cloud to arrive ...");

  while (ros::ok())
  {
    if (has_cloud_)
    {
      std::vector<Grasp> grasps;
      // detect grasps in point cloud
      if (cloud_type_ == CLOUD_GRASPS){
        std::cout << "running detect grasps for CLOUD_GRASPS " << std::endl;
        grasps = detectGraspPosesInTopicWithCandidateGrasps(*this->graspSetVec);
      } else{
        grasps = detectGraspPosesInTopic();
      }


      // visualize grasps in rviz
      if (use_rviz_)
      {
        grasps_rviz_pub_.publish(convertToVisualGraspMsg(grasps, 0.1, 0.06, 0.01, 0.02, frame_));
      }

      gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
      grasps_pub_.publish(selected_grasps_msg);


      // reset the system
      has_cloud_ = false;
      has_samples_ = false;
      has_normals_ = false;
      ROS_INFO("Waiting for point cloud to arrive ...");
    }

    ros::spinOnce();
    rate.sleep();
  }
}


std::vector<Grasp> GraspDetectionNode::detectGraspPosesInTopic()
{
  // detect grasp poses
  std::vector<Grasp> grasps;

  if (use_importance_sampling_)
  {
    cloud_camera_->filterWorkspace(workspace_);
    cloud_camera_->voxelizeCloud(0.003);
    cloud_camera_->calculateNormals(4);
    grasps = importance_sampling_->detectGrasps(*cloud_camera_);
  }
  else
  {
    // preprocess the point cloud
    grasp_detector_->preprocessPointCloud(*cloud_camera_);

    // detect grasps in the point cloud
    grasps = grasp_detector_->detectGrasps(*cloud_camera_);
  }

  // Publish the selected grasps.
  gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
  grasps_pub_.publish(selected_grasps_msg);
  ROS_INFO_STREAM("Published " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");

  return grasps;
}

std::vector<Grasp> GraspDetectionNode::detectGraspPosesInTopicWithCandidateGrasps(std::vector<GraspSet>& candidateGrasps){

  // preprocess the point cloud
  grasp_detector_->preprocessPointCloud(*cloud_camera_);
//
//  Plot plotter;
//  if (true)
//  {
//    plotter.plotFingers(candidateGrasps, cloud_camera_->getCloudProcessed(), "Candidate Grasps");
//  }

  // 3. Classify each grasp candidate. (Note: switch from a list of hypothesis sets to a list of grasp hypotheses)
  std::vector<Grasp> grasps = grasp_detector_->classifyAllGraspCandidates(*cloud_camera_, candidateGrasps);
  ROS_INFO_STREAM("Scored " << grasps.size() << " grasps.");
  return grasps;
}


std::vector<int> GraspDetectionNode::getSamplesInBall(const PointCloudRGBA::Ptr& cloud,
  const pcl::PointXYZRGBA& centroid, float radius)
{
  std::vector<int> indices;
  std::vector<float> dists;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  kdtree.radiusSearch(centroid, radius, indices, dists);
  return indices;
}


void GraspDetectionNode::cloud_callback(const sensor_msgs::PointCloud2& msg)
{
  if (!has_cloud_)
  {
    delete cloud_camera_;
    cloud_camera_ = NULL;

    Eigen::Matrix3Xd view_points(3,1);
    view_points.col(0) = view_point_;

    if (msg.fields.size() == 6 && msg.fields[3].name == "normal_x" && msg.fields[4].name == "normal_y"
      && msg.fields[5].name == "normal_z")
    {
      PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points and normals.");
    }
    else
    {
      PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points.");
    }

    has_cloud_ = true;
    frame_ = msg.header.frame_id;
  }
}


void GraspDetectionNode::cloud_indexed_callback(const gpd::CloudIndexed& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the indices at which to sample grasp candidates.
    std::vector<int> indices(msg.indices.size());
    for (int i=0; i < indices.size(); i++)
    {
      indices[i] = msg.indices[i].data;
    }
    cloud_camera_->setSampleIndices(indices);

    has_cloud_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << msg.indices.size() << " samples");
  }
}


void GraspDetectionNode::cloud_samples_callback(const gpd::CloudSamples& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the samples at which to sample grasp candidates.
    Eigen::Matrix3Xd samples(3, msg.samples.size());
    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }
    cloud_camera_->setSamples(samples);

    has_cloud_ = true;
    has_samples_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << cloud_camera_->getSamples().cols() << " samples");
  }
}

void GraspDetectionNode::cloud_grasps_callback(const gpd::CloudGrasps& msg)
{
  std::cout << "got a CloudGrasps message before if statement " << std::endl;
  if (!has_cloud_){
    std::cout << "got a CloudGrasps message " << std::endl;
    initCloudCamera(msg.cloud_sources);

    // extract the candidate grasps from the message
    std::vector<Grasp> graspVec;
    for(int i = 0; i < msg.grasps.size(); i++){
      if (use_alternate_grasp_msg_decoder_){
        std::cout << "using alternate grasp msg decoder" << std::endl;
        graspVec.push_back(this->createGraspFromGraspMsg2(msg.grasps[i]));
      }
      else{
        graspVec.push_back(this->createGraspFromGraspMsg(msg.grasps[i]));
      }

    }

    // debugging publishing
    // Publish the selected grasps.
//    gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(graspVec);
//    grasps_pub_.publish(selected_grasps_msg);
//    grasps_rviz_pub_.publish(convertToVisualGraspMsg(graspVec, 0.1, 0.06, 0.01, 0.02, frame_));

    this->graspSetVec = this->createGraspSetList(graspVec);

    std::cout << "finished creating a graspSetVec from the message" << std::endl;
    this->has_cloud_ = true;
    this->has_samples_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;
  }

}


void GraspDetectionNode::samples_callback(const gpd::SamplesMsg& msg)
{
  if (!has_samples_)
  {
    Eigen::Matrix3Xd samples(3, msg.samples.size());

    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }

    cloud_camera_->setSamples(samples);
    has_samples_ = true;

    ROS_INFO_STREAM("Received grasp samples message with " << msg.samples.size() << " samples");
  }
}


void GraspDetectionNode::initCloudCamera(const gpd::CloudSources& msg)
{
  // clean up
  delete cloud_camera_;
  cloud_camera_ = NULL;

  // Set view points.
  Eigen::Matrix3Xd view_points(3, msg.view_points.size());
  for (int i = 0; i < msg.view_points.size(); i++)
  {
    view_points.col(i) << msg.view_points[i].x, msg.view_points[i].y, msg.view_points[i].z;
  }

  std::cout << "initCloudCamera::view_points = " << view_points << std::endl;

  // Set point cloud.
  if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x"
    && msg.cloud.fields[4].name == "normal_y" && msg.cloud.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
    std::cout << "view_points:\n" << view_points << "\n";
  }
}


gpd::GraspConfigList GraspDetectionNode::createGraspListMsg(const std::vector<Grasp>& hands)
{
  gpd::GraspConfigList msg;

  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(convertToGraspMsg(hands[i]));

  msg.header.stamp = ros::Time::now();

  return msg;
}


gpd::GraspConfig GraspDetectionNode::convertToGraspMsg(const Grasp& hand)
{
  gpd::GraspConfig msg;
  tf::pointEigenToMsg(hand.getGraspBottom(), msg.bottom);
  tf::pointEigenToMsg(hand.getGraspTop(), msg.top);
  tf::pointEigenToMsg(hand.getGraspSurface(), msg.surface);
  tf::vectorEigenToMsg(hand.getApproach(), msg.approach);
  tf::vectorEigenToMsg(hand.getBinormal(), msg.binormal);
  tf::vectorEigenToMsg(hand.getAxis(), msg.axis);
  msg.width.data = hand.getGraspWidth();
  msg.score.data = hand.getScore();
  tf::pointEigenToMsg(hand.getSample(), msg.sample);

  msg.config1d_bottom = hand.getBottom();
  msg.config1d_top = hand.getTop();
  msg.config1d_center = hand.getCenter();
  msg.config1d_left = hand.getLeft();
  msg.config1d_right = hand.getRight();
  return msg;
}

FingerHand GraspDetectionNode::createFingerHandFromMsg(const gpd::FingerHand& msg){
  FingerHand finger_hand(msg.finger_width, msg.hand_outer_diameter, msg.hand_depth);

  // set all the properties of the finger hand
  finger_hand.setForwardAxis(msg.forward_axis);
  finger_hand.setLateralAxis(msg.lateral_axis);
  finger_hand.setBottom(msg.bottom);
  finger_hand.setCenter(msg.center);
  finger_hand.setLeft(msg.left);
  finger_hand.setRight(msg.right);
  finger_hand.setSurface(msg.surface);
  finger_hand.setTop(msg.top);

  // also need to set which finger placement we used

  return finger_hand;
}


Grasp GraspDetectionNode::createGraspFromGraspMsg(const gpd::GraspMsg& msg){

  FingerHand finger_hand = this->createFingerHandFromMsg(msg.finger_hand);
  double grasp_width = msg.finger_hand.hand_outer_diameter; // I don't think this is actually used for anything

  // set all the properties of the finger hand
  Eigen::Vector3d sample(msg.pose.position.x , msg.pose.position.y, msg.pose.position.z);
  Eigen::Quaternion<double> quaternion(msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z);
  Eigen::Matrix3d rotMatrix = quaternion.toRotationMatrix();
  std::cout << "x-direction  = " << rotMatrix.col(0) << std::endl;
  std::cout << "y-direction  = " << rotMatrix.col(1) << std::endl;
  std::cout << "z-direction  = " << rotMatrix.col(2) << std::endl;
  Grasp grasp(sample, rotMatrix, finger_hand, grasp_width);
  return grasp;
}

Grasp GraspDetectionNode::createGraspFromGraspMsg2(const gpd::GraspMsg& msg){

  // copy finger hand as closely as possible
  FingerHand finger_hand(msg.finger_hand.finger_width, msg.finger_hand.hand_outer_diameter, msg.finger_hand.hand_depth);
  finger_hand.setForwardAxis(msg.finger_hand.forward_axis);
  finger_hand.setLateralAxis(msg.finger_hand.lateral_axis);
  finger_hand.setBottom(msg.grasp_config.config1d_bottom);
  finger_hand.setCenter(msg.grasp_config.config1d_center);
  finger_hand.setLeft(msg.grasp_config.config1d_left);
  finger_hand.setRight(msg.grasp_config.config1d_right);
//  finger_hand.setSurface(msg.grasp_config.config1d_surface);
  finger_hand.setTop(msg.grasp_config.config1d_top);

  // set all the properties of the finger hand
  // create the Grasp object
  Eigen::Matrix3d frame;
  frame.col(0) << msg.grasp_config.approach.x, msg.grasp_config.approach.y, msg.grasp_config.approach.z;
  frame.col(1) << msg.grasp_config.binormal.x, msg.grasp_config.binormal.y, msg.grasp_config.binormal.z;
  frame.col(2) << msg.grasp_config.axis.x, msg.grasp_config.axis.y, msg.grasp_config.axis.z;

  Eigen::Vector3d sample(msg.grasp_config.sample.x, msg.grasp_config.sample.y, msg.grasp_config.sample.z);

  Grasp grasp(sample, frame, finger_hand);

  Eigen::Vector3d grasp_bottom(msg.grasp_config.bottom.x, msg.grasp_config.bottom.y, msg.grasp_config.bottom.z);
  grasp.setGraspBottom(grasp_bottom);

  Eigen::Vector3d grasp_top(msg.grasp_config.top.x, msg.grasp_config.top.y, msg.grasp_config.top.z);
  grasp.setGraspTop(grasp_top);

  Eigen::Vector3d grasp_surface(msg.grasp_config.surface.x, msg.grasp_config.surface.y, msg.grasp_config.surface.z);
  grasp.setGraspSurface(grasp_surface);


//
//  Eigen::Vector3d grasp_bottom(msg.bottom.x, msg.bottom.y, msg.bottom.z);
//  Eigen::Vector3d grasp_bottom(msg.bottom.x, msg.bottom.y, msg.bottom.z);
//  Eigen::Vector3d grasp_bottom(msg.bottom.x, msg.bottom.y, msg.bottom.z);
//  Eigen::Vector3d grasp_bottom(msg.bottom.x, msg.bottom.y, msg.bottom.z);
//  Eigen::Vector3d grasp_bottom(msg.bottom.x, msg.bottom.y, msg.bottom.z);
//  grasp.setGraspBottom()

  return grasp;
}

std::shared_ptr<std::vector<GraspSet>> GraspDetectionNode::createGraspSetList(std::vector<Grasp>& graspVec) {
  std::shared_ptr<std::vector<GraspSet>> graspSetList = std::make_shared<std::vector<GraspSet>>();

  for(int i = 0; i < graspVec.size(); i++){
    GraspSet graspSet;
    std::vector<Grasp> singleGraspVec;
    singleGraspVec.push_back(graspVec[i]);
    graspSet.setHands(singleGraspVec);
    Eigen::Array<bool, 1, Eigen::Dynamic> isValid(1);
    isValid(0,0) = true;
//    Eigen::Vector<bool> isValid(1);
//    isValid << true;
    std::cout << "isValid " << isValid << std::endl;
    graspSet.setIsValid(isValid);
    std::cout << "graspSet.getIsValid " << graspSet.getIsValid() << std::endl;
    std::cout << "graspSet.getHypotheses.size() " << graspSet.getHypotheses().size() << std::endl;
    graspSet.setSample(graspVec[i].getSample());
    graspSetList->push_back(graspSet);

  }

  return graspSetList;
}



visualization_msgs::MarkerArray GraspDetectionNode::convertToVisualGraspMsg(const std::vector<Grasp>& hands,
  double outer_diameter, double hand_depth, double finger_width, double hand_height, const std::string& frame_id)
{
  double width = outer_diameter;
  double hw = 0.5 * width;

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker left_finger, right_finger, base, approach;
  Eigen::Vector3d left_bottom, right_bottom, left_top, right_top, left_center, right_center, approach_center;

  for (int i = 0; i < hands.size(); i++)
  {
    left_bottom = hands[i].getGraspBottom() + hw * hands[i].getBinormal();
    right_bottom = hands[i].getGraspBottom() - hw * hands[i].getBinormal();
    left_top = left_bottom + hand_depth * hands[i].getApproach();
    right_top = right_bottom + hand_depth * hands[i].getApproach();

    left_center = left_bottom + 0.5 * (left_top - left_bottom) - 0.5 * finger_width * hands[i].getFrame().col(1);
    right_center = right_bottom + 0.5 * (right_top - right_bottom) + 0.5 * finger_width * hands[i].getFrame().col(1);
    approach_center = left_bottom + 0.5 * (right_bottom - left_bottom) - 0.04 * hands[i].getFrame().col(0);

    base = createHandBaseMarker(left_bottom, right_bottom, hands[i].getFrame(), 0.02, hand_height, i, frame_id);
    left_finger = createFingerMarker(left_center, hands[i].getFrame(), hand_depth, finger_width, hand_height, i*3, frame_id);
    right_finger = createFingerMarker(right_center, hands[i].getFrame(), hand_depth, finger_width, hand_height, i*3+1, frame_id);
    approach = createFingerMarker(approach_center, hands[i].getFrame(), 0.08, finger_width, hand_height, i*3+2, frame_id);

    marker_array.markers.push_back(left_finger);
    marker_array.markers.push_back(right_finger);
    marker_array.markers.push_back(approach);
    marker_array.markers.push_back(base);
  }

  return marker_array;
}


visualization_msgs::Marker GraspDetectionNode::createFingerMarker(const Eigen::Vector3d& center,
  const Eigen::Matrix3d& frame, double length, double width, double height, int id, const std::string& frame_id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time();
  marker.ns = "finger";
  marker.id = id;
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = center(0);
  marker.pose.position.y = center(1);
  marker.pose.position.z = center(2);
  marker.lifetime = ros::Duration(10);

  // use orientation of hand frame
  Eigen::Quaterniond quat(frame);
  marker.pose.orientation.x = quat.x();
  marker.pose.orientation.y = quat.y();
  marker.pose.orientation.z = quat.z();
  marker.pose.orientation.w = quat.w();

  // these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length; // forward direction
  marker.scale.y = width; // hand closing direction
  marker.scale.z = height; // hand vertical direction

  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.5;

  return marker;
}


visualization_msgs::Marker GraspDetectionNode::createHandBaseMarker(const Eigen::Vector3d& start,
  const Eigen::Vector3d& end, const Eigen::Matrix3d& frame, double length, double height, int id,
  const std::string& frame_id)
{
  Eigen::Vector3d center = start + 0.5 * (end - start);

  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time();
  marker.ns = "hand_base";
  marker.id = id;
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = center(0);
  marker.pose.position.y = center(1);
  marker.pose.position.z = center(2);
  marker.lifetime = ros::Duration(10);

  // use orientation of hand frame
  Eigen::Quaterniond quat(frame);
  marker.pose.orientation.x = quat.x();
  marker.pose.orientation.y = quat.y();
  marker.pose.orientation.z = quat.z();
  marker.pose.orientation.w = quat.w();

  // these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length; // forward direction
  marker.scale.y = (end - start).norm(); // hand closing direction
  marker.scale.z = height; // hand vertical direction

  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 1.0;

  return marker;
}


int main(int argc, char** argv)
{
  // seed the random number generator
  std::srand(std::time(0));

  // initialize ROS
  ros::init(argc, argv, "detect_grasps");
  ros::NodeHandle node("~");

  GraspDetectionNode grasp_detection(node);
  grasp_detection.run();

  return 0;
}
