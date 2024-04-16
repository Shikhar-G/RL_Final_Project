

#include "Informed_RRT.h"
#include "shared/math/geometry.h"
#include "gflags/gflags.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "glog/logging.h"
#include "ros/ros.h"
#include "ros/package.h"
#include <algorithm>
#include <vector>



namespace planner {
    Informed_RRT_Star::Informed_RRT_Star(){}

    Informed_RRT_Star::Informed_RRT_Star(vector_map::VectorMap* map) {
        vector_map_ = map;
        // Find min/max x and y
        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        for (const auto& line : vector_map_->lines)
        {
            min_x = std::min(min_x, std::min(line.p0.x(), line.p1.x()));
            max_x = std::max(max_x, std::max(line.p0.x(), line.p1.x()));
            min_y = std::min(min_y, std::min(line.p0.y(), line.p1.y()));
            max_y = std::max(max_y, std::max(line.p0.y(), line.p1.y()));
        }
        assert (min_x != std::numeric_limits<float>::infinity());
        assert (max_x != -std::numeric_limits<float>::infinity());
        assert (min_y != std::numeric_limits<float>::infinity());
        assert (max_y != -std::numeric_limits<float>::infinity());
        min_x_ = min_x;
        max_x_ = max_x;
        min_y_ = min_y;
        max_y_ = max_y;
        // ROS_INFO("min_x: %f, max_x: %f, min_y: %f, max_y: %f", min_x_, max_x_, min_y_, max_y_);
    }
    
    bool Informed_RRT_Star::Plan() {
        // Reset the tree
        ROS_INFO("Planning...");
        // ROS_INFO("Size of point cloud: %ld", point_cloud_.size());
        this->Clear();
        goal_index_ = -1;
        bool goal_reached = false;
        // Add the start node
        vertices_.push_back(start_);
        parents_.push_back(0);
        costs_.push_back(0);
        costs_distance_.push_back(0);
        float curr_radius = 0.5;
        Eigen::Vector2f center = start_;
        step_size_ = 0.5;
        // iterate through the number of iterations, but don't stop if the goal hasn't been reached
        for (size_t i = 0, true_iter = 1; i < num_iterations_ && true_iter <= max_iterations_; true_iter++)
        {
            Eigen::Vector2f random_point;
            // Sample a random point

            if (goal_reached)
            {
                float c_min = (goal_ - start_).norm();
                Eigen::Matrix3f rotation_to_world = RotationToWorldFrame(start_,goal_, c_min);
                random_point = InformedSampleRandomPoint(c_min, goal_index_, rotation_to_world);
                i++;
            }
            else if (true_iter % 500 == 0) {
                random_point = goal_;
                if (center == start_) center = goal_;
                else center = start_;

                if (true_iter % 10000 == 0)
                {
                    step_size_ /= 1.5;
                }
                // ROS_INFO("iteration: %ld", true_iter);
            }
            else {
                if (true_iter % 50 == 0)
                {
                    curr_radius *= 1.5;
                }
                random_point = SampleRandomPoint(curr_radius, center);
            }
            // Make sure the random point is not too close to any wall
            if (!IsValidVertex(random_point))
            {
                continue;
            }
            // Find the nearest vertex
            size_t nearest_vertex_index = FindNearestVertex(random_point);
            // Steer towards the random point
            Eigen::Vector2f new_vertex = Steer(nearest_vertex_index, random_point);
            // Check if the new vertex is valid
            float closest_distance = ClosestDistanceToWall(vertices_[nearest_vertex_index], new_vertex);
            if (!IsCollision(closest_distance))
            {
                // Find the nearest vertex to the new vertex
                vector<size_t> vertices_in_radius = FindVerticesInRadius(new_vertex, radius_);
                size_t nearest_vertex_new = FindNearestVertexInRadius(new_vertex, nearest_vertex_index, vertices_in_radius);
                // Add the new vertex to the tree
                vertices_.push_back(new_vertex);
                parents_.push_back(nearest_vertex_new);
                float closest_distance_nearest = ClosestDistanceToWall(vertices_[nearest_vertex_new], new_vertex);
                // float closest_distance_nearest = 0;
                costs_.push_back(costs_[nearest_vertex_new] + Cost(vertices_[nearest_vertex_new], new_vertex, closest_distance_nearest));
                costs_distance_.push_back(costs_distance_[nearest_vertex_new] + (vertices_[nearest_vertex_new] - new_vertex).norm());
                // // check if nearest_vertex_new is in edges_
                // if (edges_.find(nearest_vertex_new) == edges_.end())
                // {
                //     edges_[nearest_vertex_new] = vector<size_t>();
                // }
                // Add the new vertex to the edges
                // edges_[nearest_vertex_new].push_back(vertices_.size() - 1);
                // Rewire the tree
                Rewire(new_vertex, vertices_in_radius);
                // Check if the goal is reached
                if ((new_vertex - goal_).norm() < goal_radius_)
                {
                    // Set the goal index if goal is not reached yet or the new vertex has a lower cost
                    if (!goal_reached || costs_.back() < costs_[goal_index_]) {
                        // if (!goal_reached) {
                        //     ROS_INFO("Goal reached at iteration %zu", true_iter );
                        // }
                        goal_index_ = vertices_.size() - 1;
                        goal_reached = true;
                    }
                }
            }
        }
        return goal_reached;
    }

    bool Informed_RRT_Star::IsValidVertex(const Eigen::Vector2f& vertex)
    {
        float closest_distance = ClosestDistanceToWall(vertex, vertex);
        return !IsCollision(closest_distance);
    }

    bool Informed_RRT_Star::isPathValid(size_t curr_index)
    {
        // only check 3
        for (size_t i = curr_index; i < path_.size() - 1 && i < curr_index + 2; i++)
        {
            float closest_distance = ClosestDistanceToWall(path_[i], path_[i+1]);
            if (IsCollision(closest_distance))
            {
                return false;
            }
        }
        return true;
    }

    vector<Eigen::Vector2f> Informed_RRT_Star::GetPath(bool smooth)
    {
        vector<Eigen::Vector2f> path_out;
        vector<Eigen::Vector2f> smooth_path;
        // the goal can not be larger then the size of vertices_
        assert(!(vertices_.size() < goal_index_));

        size_t curr_vertex = goal_index_;
        //push goal to path.
        path_out.push_back(vertices_[curr_vertex]);

        //starting from the goal work backwards to the start
        while (curr_vertex != 0)
        {
            size_t parent_vertex = parents_[curr_vertex];
            // Eigen::Vector2f curr = vertices_[curr_vertex];
            // Eigen::Vector2f parent = vertices_[parent_vertex];
            curr_vertex = parent_vertex;
            path_out.push_back(vertices_[curr_vertex]);
        }
        
        //the path goes from goal to start right now, reverse it
        std::reverse(path_out.begin(),path_out.end());
        path_out.push_back(goal_);
        if (!smooth)
        {
            path_ = path_out;
            return path_out;
        }
        smooth_path.resize(path_out.size());
        size_t position_index = 0;
        while(position_index < path_out.size())
        {
            Eigen::Vector2f evaluating_vertex = path_out[position_index];
            smooth_path[position_index] = evaluating_vertex;
            for(size_t i=path_out.size() - 1; i > position_index; i--)
            {
                float closest_distance = ClosestDistanceToWall(evaluating_vertex, path_out[i]);
                if(!IsCollision(closest_distance) && Cost(evaluating_vertex, path_out[i], closest_distance) < costs_[i] - costs_[position_index])
                {
                    smooth_path[i] = path_out[i];
                    float num_points_between = i - position_index;
                    if (num_points_between > 1)
                    {
                        float x_step = (path_out[i][0] - evaluating_vertex[0])/(num_points_between);
                        float y_step = (path_out[i][1] - evaluating_vertex[1])/(num_points_between);
                        for (size_t j = 1; j < num_points_between; j++)
                        {
                            smooth_path[position_index + j] = Eigen::Vector2f(evaluating_vertex[0] + j*x_step, evaluating_vertex[1] + j*y_step);
                        }
                    }
                    position_index = i;
                    break;
                }
            }
            position_index++;
        }
        path_ = smooth_path;
        return smooth_path;
    }


    vector<Eigen::Vector2f> Informed_RRT_Star::GetPath()
    {
        vector<Eigen::Vector2f> path_out;
        vector<size_t> path_indices;
        vector<Eigen::Vector2f> smooth_path;
        // the goal can not be larger then the size of vertices_
        assert(!(vertices_.size() < goal_index_));

        size_t curr_vertex = goal_index_;
        //push goal to path.
        path_out.push_back(vertices_[curr_vertex]);
        path_indices.push_back(curr_vertex);

        //starting from the goal work backwards to the start
        while (curr_vertex != 0)
        {
            size_t parent_vertex = parents_[curr_vertex];
            // Eigen::Vector2f curr = vertices_[curr_vertex];
            // Eigen::Vector2f parent = vertices_[parent_vertex];
            curr_vertex = parent_vertex;
            path_out.push_back(vertices_[curr_vertex]);
            path_indices.push_back(curr_vertex);
        }
        
        //the path goes from goal to start right now, reverse it
        std::reverse(path_out.begin(),path_out.end());
        std::reverse(path_indices.begin(),path_indices.end());
        // add the goal as a vertex
        vertices_.push_back(goal_);
        parents_.push_back(goal_index_);
        float closest_distance = ClosestDistanceToWall(vertices_[goal_index_], goal_);
        costs_.push_back(costs_[goal_index_] + Cost(vertices_[goal_index_], goal_, closest_distance));
        path_out.push_back(goal_);
        path_indices.push_back(vertices_.size() - 1);

        //path smoothing
        // initialize the smooth path with the start
        smooth_path.resize(path_out.size());
        size_t position_index = 0;
        while(position_index < path_out.size())
        {
            // ROS_INFO("position_index: %zu", position_index);
            Eigen::Vector2f evaluating_vertex = path_out[position_index];
            smooth_path[position_index] = evaluating_vertex;
            for(size_t i=path_out.size() - 1; i > position_index; i--)
            {
                size_t start = path_indices[position_index];
                size_t end = path_indices[i];
                float closest_distance = ClosestDistanceToWall(evaluating_vertex, path_out[i]);
                if(!IsCollision(closest_distance, safety_margin_) && Cost(evaluating_vertex, path_out[i], closest_distance) < costs_[end] - costs_[start])
                {
                    // ROS_INFO("Closest distance between %zu and %zu: %f", start, end, closest_distance);
                    smooth_path[i] = path_out[i];
                    float num_points_between = i - position_index;
                    if (num_points_between > 1)
                    {
                        float x_step = (path_out[i][0] - evaluating_vertex[0])/(num_points_between);
                        float y_step = (path_out[i][1] - evaluating_vertex[1])/(num_points_between);
                        for (size_t j = 1; j < num_points_between; j++)
                        {
                            smooth_path[position_index + j] = Eigen::Vector2f(evaluating_vertex[0] + j*x_step, evaluating_vertex[1] + j*y_step);
                        }
                    }
                    // ROS_INFO("smoothed path from %zu to %zu", position_index, i);
                    float cost_reduction = (costs_[end] - costs_[start]) - (Cost(evaluating_vertex, path_out[i], closest_distance));
                    // ROS_INFO("cost reduction: %f", cost_reduction);
                    for (size_t j = path_out.size() - 1; j >= i; j--)
                    {
                        costs_[path_indices[j]] -= cost_reduction;
                    }
                    position_index = i;
                    break;
                }
            }
            position_index++;
        }
        path_ = smooth_path;
        return smooth_path;
    }

    float Informed_RRT_Star::Cost(const Eigen::Vector2f& start, const Eigen::Vector2f& end, float closest_distance)
    {
        return (start - end).norm() + 1/closest_distance * 5;
    }

    Eigen::Vector3f Informed_RRT_Star::SampleUnitBall()
    {
        while(true)
        {
            float x = rng_.UniformRandom(-1, 1);
            float y = rng_.UniformRandom(-1, 1);

            if(Sq(x) + Sq(y) <= 1)
            {
                return Eigen::Vector3f(x,y,0.0);
            }
        }

    }
    Eigen::Matrix3f Informed_RRT_Star::RotationToWorldFrame(const Eigen::Vector2f& start, const Eigen::Vector2f& goal, float L)
    {
        Eigen::Vector3f a1((goal[0]-start[0])/L, (goal[1]-start[1])/L, 0.0);
        Eigen::Vector3f e1(1,0,0);
        Eigen::Matrix3f M = a1 * e1.transpose();
        Eigen::Matrix3f diag;
        float det;
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V_T = svd.matrixV();
        Eigen::Matrix3f V_T_trans = V_T.transpose();
        det = U.determinant()*V_T_trans.determinant();
        diag << 1, 0, 0,
                0, 1, 0,
                0, 0, det;
        Eigen::Matrix3f C = U * diag * V_T;

        return C;

    }

    Eigen::Vector2f Informed_RRT_Star::SampleRandomPoint(float radius, const Eigen::Vector2f& center)
    {
        // Sample a random point with given radius of the start point but not less than min_x_ and max_x_
        float min_x = std::max(min_x_, center[0] - radius);
        float max_x = std::min(max_x_, center[0] + radius);
        float min_y = std::max(min_y_, center[1] - radius);
        float max_y = std::min(max_y_, center[1] + radius);
        float x = rng_.UniformRandom(min_x, max_x);
        float y = rng_.UniformRandom(min_y, max_y);
        // Sample a random point within the bounds

        // float x = rng_.UniformRandom(min_x_, max_x_);
        // float y = rng_.UniformRandom(min_y_, max_y_);
        // double x = min_x_ + static_cast<double>(rand()) / RAND_MAX * (max_x_ - min_x_);
        // double y = min_y_ + static_cast<double>(rand()) / RAND_MAX * (max_y_ - min_y_);
        return Eigen::Vector2f(x, y);
    }

    Eigen::Vector2f Informed_RRT_Star::InformedSampleRandomPoint(float c_min, size_t goal_point_index, Eigen::Matrix3f rotation_to_world)
    {
        
        float c_max = costs_[goal_point_index];
        Eigen::Vector2f center = (start_ - goal_)/2;
        Eigen::Vector3f center_vec(center[0],center[1],0);
        float r_1 = c_max/2;
        float r_2 = (Sq(c_max) - Sq(c_min))/2;
        // Eigen::Vector3f r(r_1,r_2,r_2);
        Eigen::DiagonalMatrix<float,3> L(r_1,r_2,r_2);
        Eigen::Matrix3f C = rotation_to_world;
        Eigen::Vector3f ball = SampleUnitBall();
        Eigen::Vector3f rand = C*L*ball + center_vec;

        return Eigen::Vector2f(rand[0], rand[1]);
    }

    size_t Informed_RRT_Star::FindNearestVertex(const Eigen::Vector2f& point)
    {
        // Find the nearest vertex
        size_t nearest_vertex_index = vertices_.size();
        double min_cost = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < vertices_.size(); i++)
        {
            double cost = (vertices_[i] - point).norm();
            if (cost < min_cost)
            {
                nearest_vertex_index = i;
                min_cost = cost;
            }
        }
        return nearest_vertex_index;
    }

    Eigen::Vector2f Informed_RRT_Star::Steer(size_t nearest_vertex_index, const Eigen::Vector2f& random_point)
    {
        // Steer towards the random point
        Eigen::Vector2f nearest = vertices_[nearest_vertex_index];
        Eigen::Vector2f direction = (random_point - nearest).normalized();
        return nearest + direction * step_size_;
    }

    bool Informed_RRT_Star::IsCollision(float closest_distance, float margin)
    {
        return closest_distance < margin;
    }

    bool Informed_RRT_Star::IsCollision(float closest_distance)
    {
        return closest_distance < safety_margin_;
    }
    // {
    //     // check if lines are within safety margin of each other or intersect
    //     for (const auto& line : vector_map_->lines)
    //     {
    //         if (line.CloserThan(start, end, safety_margin_))
    //         {
    //             return true;
    //         }
    //     }
    //     return false;
    // }

    float Informed_RRT_Star::ClosestDistanceToWall(const Eigen::Vector2f& start, const Eigen::Vector2f& end)
    {
        float min_distance = std::numeric_limits<float>::infinity();
        // length of the line segment square
        for (const auto& wall : vector_map_->lines)
        {
            if (wall.CloserThan(start, end, safety_margin_))
            {
                return 0;
            }
            else {
                min_distance = std::min(min_distance, wall.ClosestApproach(start, end));
            }
        }
        // iterate through point cloud and find the closest distance to a point
        for (const Eigen::Vector2f& point : point_cloud_)
        {
            // float distance;
            // if (start == end)
            //     distance = (point - start).norm();
            // else {
            //     // compute distance from point to line manually
            //     Eigen::Vector2f start_to_point = point - start;
            //     Eigen::Vector2f start_to_end = end - start;
            //     float t = start_to_point.dot(start_to_end) / start_to_end.squaredNorm();
            //     if (t < 0)
            //     {
            //         distance = start_to_point.norm();
            //     }
            //     else if (t > 1)
            //     {
            //         distance = (point - end).norm();
            //     }
            //     else {
            //         distance = (point - (start + t * start_to_end)).norm();
            //     }
            // }
            // if (distance < safety_margin_)
            // {
            //     return 0;
            // }
            // else {
            //     min_distance = std::min(distance, min_distance);
            // }
            float sq_dist;
            Eigen::Vector2f projected_point;
            geometry::ProjectPointOntoLineSegment<float>(point, start, end, &projected_point, &sq_dist);
            if (sq_dist < Sq(safety_margin_ / 2))
            {
                return 0;
            }
            else {
                min_distance = std::min(sqrt(sq_dist), min_distance);
            }
        }
        return min_distance;
    }




    vector<size_t> Informed_RRT_Star::FindVerticesInRadius(const Eigen::Vector2f& point, double radius)
    {
        vector<size_t> vertices_in_radius;
        for (size_t i = 0; i < vertices_.size(); i++)
        {
            if ((vertices_[i] - point).norm() < radius)
            {
                vertices_in_radius.push_back(i);
            }
        }
        return vertices_in_radius;
    }

    size_t Informed_RRT_Star::FindNearestVertexInRadius(const Eigen::Vector2f& point, size_t nearest_vertex_index, const vector<size_t>& vertices_in_radius)
    {
        double min_cost = costs_[nearest_vertex_index];
        for (size_t i : vertices_in_radius)
        {
            float closest_distance = ClosestDistanceToWall(vertices_[i], point); 
            if (!IsCollision(closest_distance))
            {
                double cost = costs_[i] + Cost(vertices_[i], point, closest_distance);
                if (cost < min_cost)
                {
                    nearest_vertex_index = i;
                    min_cost = cost;
                }
            }
        }
        return nearest_vertex_index;
    }


    void Informed_RRT_Star::Rewire(const Eigen::Vector2f& new_vertex,const vector<size_t>& vertices_in_radius)
    {
        size_t new_vertex_index = costs_.size() - 1;
        float new_vertex_cost = costs_[new_vertex_index];
        float new_vertex_cost_distance = costs_distance_[new_vertex_index];
        for(size_t vertex_near : vertices_in_radius)
        {
            float closest_distance_near_new = ClosestDistanceToWall(vertices_[vertex_near], new_vertex);
            if(!IsCollision(closest_distance_near_new))
            {
                float cost = new_vertex_cost + Cost(new_vertex, vertices_[vertex_near], closest_distance_near_new);
                float cost_distance = new_vertex_cost_distance + (new_vertex - vertices_[vertex_near]).norm();
                // Make sure new edges would not collide with any wall
                if(cost < costs_[vertex_near])
                {
                    costs_[vertex_near] = cost;
                    costs_distance_[vertex_near] = cost_distance;
                    // edges_[new_vertex_index].push_back(vertex_near);
                    // edges_[vertex_near].erase(std::remove(edges_[vertex_near].begin(), edges_[vertex_near].end(),  parents_[vertex_near]), edges_[vertex_near].end());
                    parents_[vertex_near] = new_vertex_index;
                }
            }

        }
    }

    vector<pair<Eigen::Vector2f, Eigen::Vector2f>> Informed_RRT_Star::GetTree()
    {
        std::vector<pair<Eigen::Vector2f, Eigen::Vector2f>> tree;
        for (size_t i = 0; i < vertices_.size(); i++)
        {
            size_t parent = parents_[i];
            if (parent != i)
            {
                tree.push_back({vertices_[i], vertices_[parent]});
            }
        }
        return tree;
    }
    
}
