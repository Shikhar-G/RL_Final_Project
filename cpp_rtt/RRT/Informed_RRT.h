//this will be dedicated to the 




#include <algorithm>
#include <vector>
#include <unordered_map>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "shared/math/line2d.h"
#include "shared/util/random.h"
#include "vector_map/vector_map.h"
#include "shared/util/random.h"
#include "global_planner.h"

#include "simple_queue.h"

//ros testing
#include "ros/ros.h"
#include "ros/package.h"

#ifndef INFORMED_RRT_H_
#define INFORMED_RRT_H_

namespace planner {

using std::vector;
using std::unordered_map;

// General parent class planner that will be inherited by all other planners




class Informed_RRT_Star : public Planner{

    private:
        Eigen::Vector2f start_;
        float start_angle_;
        Eigen::Vector2f goal_;
        //the goal radius is in meters, creates 
        float goal_radius_ = 0.5;
        vector<Eigen::Vector2f> vertices_;
        unordered_map<size_t, vector<size_t>> edges_;
        vector<size_t> parents_;
        vector<float> costs_;
        vector<float> costs_distance_;
        vector<Eigen::Vector2f> path_;
        size_t goal_index_ = -1;
        vector_map::VectorMap *vector_map_;
        std::vector<Eigen::Vector2f> point_cloud_;
        size_t num_iterations_ = 5000;
        size_t max_iterations_ = 100000;
        float min_x_;
        float max_x_;
        float min_y_;
        float max_y_;
        float radius_ = 1;
        float step_size_ = 0.5;
        float safety_margin_ = 0.2;
        //random number generator
        util_random::Random rng_;

        Eigen::Vector2f SampleRandomPoint(float radius, const Eigen::Vector2f& center);
        Eigen::Vector2f InformedSampleRandomPoint(float c_min, size_t goal_point_index, Eigen::Matrix3f rotation_to_world);
        size_t FindNearestVertex(const Eigen::Vector2f& point);
        Eigen::Vector2f Steer(size_t nearest_vertex_index, const Eigen::Vector2f& random_point);
        bool IsValidVertex(const Eigen::Vector2f& vertex);
        bool IsCollision(float closest_distance);
        bool IsCollision(float closest_distance, float margin);
        void Rewire(const Eigen::Vector2f& new_vertex, const vector<size_t>& nearby_vertices);
        float Cost(const Eigen::Vector2f& start, const Eigen::Vector2f& end, float closest_distance);
        size_t FindNearestVertexInRadius(const Eigen::Vector2f& point, size_t nearest_vertex_index, const vector<size_t>& vertices_in_radius);
        vector<size_t> FindVerticesInRadius(const Eigen::Vector2f& point, double radius);
        float ClosestDistanceToWall(const Eigen::Vector2f& start, const Eigen::Vector2f& end);
        //informed rrt sampling
        Eigen::Vector3f SampleUnitBall();
        Eigen::Matrix3f RotationToWorldFrame(const Eigen::Vector2f& start, const Eigen::Vector2f& goal, float L);

        
    public:
        //initialization
        Informed_RRT_Star();
        Informed_RRT_Star(vector_map::VectorMap* map);
        Informed_RRT_Star(Eigen::Vector2f start, Eigen::Vector2f goal): start_(start), goal_(goal){}


        
        void Clear() {vertices_.clear(); edges_.clear(); parents_.clear(); costs_.clear(); goal_index_ = -1;}

        void SetStart(Eigen::Vector2f start) {start_ = start;}
        void SetStart(Eigen::Vector2f start, float start_angle);
        void SetGoal(Eigen::Vector2f goal) {goal_ = goal;}
        void SetPointCloud(const std::vector<Eigen::Vector2f>& point_cloud) {point_cloud_ = point_cloud;}
        bool isPathValid(size_t curr_index);

        bool Plan();

        vector<pair<Eigen::Vector2f, Eigen::Vector2f>> GetTree();
        vector<Eigen::Vector2f> GetPath();
        vector<Eigen::Vector2f> GetPath(bool smooth);

    };
}
#endif //GLOBAL_PLANNER_H_
