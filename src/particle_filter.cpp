/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random> // Need this for sampling from distributions

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    weights.resize(num_particles);

    // This line creates a normal (Gaussian) distribution for x.
    normal_distribution<double> dist_x(x, std[0]);

    // This line creates a normal (Gaussian) distribution for y.
    normal_distribution<double> dist_y(y, std[1]);

    // This line creates a normal (Gaussian) distribution for theta.
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int I=0; I<num_particles; I++) {
        particles.push_back({
            I,
            dist_x(gen),
            dist_y(gen),
            dist_theta(gen),
            1.0
        });
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    const double yaw_d_t = yaw_rate * delta_t;
    const double vel_yaw = velocity / yaw_rate;

    for (int i = 0; i < num_particles; i++){

        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else {
            double theta_updated = particles[i].theta + yaw_d_t;
            particles[i].x += vel_yaw * (sin(theta_updated) - sin(particles[i].theta));
            particles[i].y += vel_yaw * (-cos(theta_updated) + cos(particles[i].theta));
            particles[i].theta = theta_updated;
        }

        // Add random Gaussian noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    const double gaussian_first_term = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    const double gaussian_denom_x = 2 * std_landmark[0]*std_landmark[0];
    const double gaussian_denom_y = 2 * std_landmark[1]*std_landmark[1];

    // for each particle
    for (int i = 0; i < num_particles; i++){
        // pre-compute to speed up transformation caclulation
        const double sin_theta = sin(particles[i].theta);
        const double cos_theta = cos(particles[i].theta);

        // For calculating multi-variate Gaussian distribution of each observation, for each particle
        double multi_variate_gaussian_distribution = 1.0;

        // for each observation
        for (int j = 0; j < observations.size(); j++){

            // Perform observation measurement transformations (rotation and translation)
            double transformed_observation_x =
                    particles[i].x + (observations[j].x * cos_theta) - (observations[j].y * sin_theta);
            double transformed_observation_y =
                    particles[i].y + (observations[j].x * sin_theta) + (observations[j].y * cos_theta);

            // Find nearest landmark
            vector<double> landmark_obs_dist (map_landmarks.landmark_list.size());
            for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {

                // Only search for landmarks in sensor range of the particle
                // If in range, put in the distance vector for calculating nearest neighbor
                double landmark_part_dist = sqrt(
                        pow(particles[i].x - map_landmarks.landmark_list[k].x_f, 2) +
                        pow(particles[i].y - map_landmarks.landmark_list[k].y_f, 2));

                if (landmark_part_dist <= sensor_range) {
                    landmark_obs_dist[k] = sqrt(
                            pow(transformed_observation_x - map_landmarks.landmark_list[k].x_f, 2) +
                            pow(transformed_observation_y - map_landmarks.landmark_list[k].y_f, 2));
                } else {
                    // Fill those out of range with very large number
                    landmark_obs_dist[k] = 1000000;
                }
            }

            // Associate the observation point with its nearest landmark neighbor
            int min_pos = distance(landmark_obs_dist.begin(),min_element(landmark_obs_dist.begin(),landmark_obs_dist.end()));
            float m_x = map_landmarks.landmark_list[min_pos].x_f;
            float m_y = map_landmarks.landmark_list[min_pos].y_f;

            // Calculate multi-variate Gaussian distribution
            double x_diff = transformed_observation_x - m_x;
            double y_diff = transformed_observation_y - m_y;
            double gaussian_second_term = ((x_diff * x_diff) / gaussian_denom_x) + ((y_diff * y_diff) / gaussian_denom_y);

            // append to multi_variate_gaussian_distribution as a product
            multi_variate_gaussian_distribution *= gaussian_first_term * exp(-gaussian_second_term);
        }

        particles[i].weight = multi_variate_gaussian_distribution;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> new_particles;

    for (int i = 0; i < num_particles; i++) {
        new_particles.push_back(particles[dist_particles(gen)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
