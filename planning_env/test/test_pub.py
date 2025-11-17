#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import os

def auto_topic_publisher():
    # 1. 初始化匿名节点（节点名自动生成，如"publisher_12345"）
    rospy.init_node("auto_publisher", anonymous=True)
    
    # 2. 获取当前节点的完整名称（包含随机后缀）
    node_name = rospy.get_name()  # 格式："/auto_publisher_12345"
    
    # 3. 自动生成话题名：从节点名中提取后缀，拼接为话题名
    #   示例：节点名"/auto_publisher_12345" → 话题名"/chatter_12345"
    node_suffix = node_name.split("_")[-1]  # 提取"12345"
    topic_name = f"chatter_{node_suffix}"
    
    # 4. 创建发布者
    pub = rospy.Publisher(topic_name, String, queue_size=10)
    rate = rospy.Rate(1)  # 1Hz发布频率
    
    while not rospy.is_shutdown():
        msg = f"Auto topic: {topic_name} (node: {node_name})"
        pub.publish(msg)
        rospy.loginfo(msg)
        rate.sleep()

if __name__ == "__main__":
    auto_topic_publisher()