"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random
import math


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.
    result_point = []
    sample_point_names = {}
    already_taken_list = []  # list to store all the traversed point pairs
    max_count = math.factorial(len(input_points)) / (2 * (math.factorial(len(input_points) - 2)))

    def choose_sample():
        while (len(already_taken_list) <= max_count):
            sample_points = random.sample(input_points, k=2)
            sample_point_names = sample_points[0].get("name"), sample_points[1].get("name")

            if ((sample_point_names in already_taken_list) == True) or (
                    (sample_point_names[::-1] in already_taken_list) == True):  # checking duplicate point pairs
                continue
            already_taken_list.append(sample_point_names)
            return (sample_points)

    for x in range(k):
        inlier_list = [];
        outlier_list = [];
        result = {}
        sample_points = choose_sample()
        sample_point_names = sample_points[0].get("name"), sample_points[1].get("name")
        p1 = sample_points[0].get("value")
        p2 = sample_points[1].get("value")
        rest_points = [n for n in input_points if n not in sample_points]
        distance_sum = 0
        for i in range(len(rest_points)):

            p3 = rest_points[i].get("value")  # the point from which distance will be calculated

            if ((p2[0] - p1[0]) != 0):
                m = (p2[1] - p1[1]) / (p2[0] - p1[0]) # calculating slope of the line
                # calculating intercept
                b = p1[1] - (m * p1[0])
                # distance calculation
                distance = abs((m * p3[0] - p3[1] + b) / math.sqrt((m * m) + 1))
            else:
                distance = abs(p3[0] - p2[0])
            if distance <= t:
                inlier = rest_points[i].get("name")
                inlier_list.append(inlier)
                distance_sum += distance
            else:
                outlier = rest_points[i].get("name")
                outlier_list.append(outlier)

            inlier_count = len(inlier_list)
            outlier_count = len(outlier_list)
            if (inlier_count >= d):
                distance_avg = distance_sum / inlier_count

                result["inlier_list"] = (list(sample_point_names) + inlier_list)
                result["Outlier_list"] = outlier_list
                result["Inlier Count"] = inlier_count
                result["Outlier Count"] = outlier_count
                result["Avg Distance"] = distance_avg

                result_point.append(result)
        if len(already_taken_list) == max_count:
            break
    distance_only = [x['Avg Distance'] for x in result_point]

    # finding point pair with minimum distance
    for k in range(len(result_point)):
        if (result_point[k].get("Avg Distance") == (min(distance_only))):
            return sorted(result_point[k].get("inlier_list")), sorted(result_point[k].get("Outlier_list"))
            break

    # raise NotImplementedError


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()
