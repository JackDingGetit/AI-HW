import pybullet as p
import pybullet_data
import time
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=1)

# 创建小物块
cube_size = 0.1
cube_pos = [0.775, 0, cube_size/2]
cube_id = p.loadURDF("cube_small.urdf", cube_pos, useFixedBase=0)

# 约束ID（用于绑定物块和夹爪）
grasp_constraint = None

# 机械臂和夹爪的关节索引
arm_joints = [0, 1, 2, 3, 4, 5, 6]
finger_joints = [9, 10]

# 打印关节信息帮助调试
print("=== 机械臂关节信息 ===")
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"关节 {i}: {joint_info[1].decode('utf-8')}")

def move_to_pose(robot_id, target_pos, target_ori=None, steps=200):
    """
    移动机械臂到指定位置和姿态
    target_ori: 目标姿态（四元数），如果为None则只控制位置
    """
    if target_ori is None:
        joint_positions = p.calculateInverseKinematics(robot_id, 11, target_pos)
    else:
        joint_positions = p.calculateInverseKinematics(robot_id, 11, target_pos, targetOrientation=target_ori)
    
    for i in range(steps):
        for j, idx in enumerate(arm_joints):
            p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, targetPosition=joint_positions[j])
        p.stepSimulation()
        time.sleep(1/240.)

def control_gripper(robot_id, open_width, steps=100, force=50):
    """控制夹爪开合，open_width范围约0-0.04"""
    for i in range(steps):
        for idx in finger_joints:
            p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, 
                                   targetPosition=open_width, force=force)
        p.stepSimulation()
        time.sleep(1/240.)

def grasp_object(robot_id, cube_id):
    """使用约束强制抓取物块"""
    global grasp_constraint
    # 创建约束：将物块绑定到夹爪末端（关节11）
    grasp_constraint = p.createConstraint(
        robot_id, 11,          # 父物体：机械臂
        cube_id, -1,           # 子物体：物块
        p.JOINT_FIXED,         # 固定约束
        [0, 0, 0],             # 本地位置偏移（父）
        [0, 0, 0],             # 本地位置偏移（子）- 物块中心与夹爪对齐
        [0, 0, 0]              # 世界坐标系中的位置
    )
    print(f"  [约束已创建] 物块绑定到夹爪")

def release_object():
    """释放物块"""
    global grasp_constraint
    if grasp_constraint is not None:
        p.removeConstraint(grasp_constraint)
        grasp_constraint = None
        print(f"  [约束已移除] 物块已释放")

# 定义常用的夹子姿态（四元数）
# 正下方（夹子向下）
down_ori = p.getQuaternionFromEuler([math.pi, 0, 0])
# 水平向前（夹子水平）
forward_ori = p.getQuaternionFromEuler([math.pi/2, 0, 0])
# 45度倾斜
tilted_ori = p.getQuaternionFromEuler([math.pi*0.75, 0, 0])

# ====== 抓取演示流程 ======
time.sleep(1)

# 1. 张开夹爪
print("1. 张开夹爪...")
control_gripper(robot_id, 0.04)  # 0.04是合理的最大张开宽度

# 2. 移动到物块上方，设置夹子向下的姿态
print("2. 移动到物块上方（夹子向下）...")
move_to_pose(robot_id, [cube_pos[0], cube_pos[1], cube_pos[2] + 0.3], target_ori=down_ori)

# 3. 下降到抓取位置（物块高度一半的位置）
print("3. 下降到抓取位置...")
grab_height = cube_pos[2]  # 物块中心高度（即高度一半）
move_to_pose(robot_id, [cube_pos[0], cube_pos[1], grab_height], target_ori=down_ori)

# 4. 闭合夹爪 + 创建约束绑定物块
print("4. 闭合夹爪并绑定物块...")
control_gripper(robot_id, 0.0)
time.sleep(0.3)
grasp_object(robot_id, cube_id)  # 强制绑定物块到夹爪

# 5. 提升物块
print("5. 提升物块...")
move_to_pose(robot_id, [cube_pos[0], cube_pos[1], cube_pos[2] + 0.4], target_ori=down_ori)

# 6. 移动到目标位置上方（左侧）
target_pos = [0, 0.775, cube_size/2]
print("6. 移动到目标位置...")
move_to_pose(robot_id, [target_pos[0], target_pos[1], target_pos[2] + 0.4], target_ori=down_ori)

# 7. 下降到放置位置
print("7. 下降到放置位置...")
drop_height = target_pos[2] + 0.02  # 物块半径 + 一点间隙
move_to_pose(robot_id, [target_pos[0], target_pos[1], drop_height], target_ori=down_ori)

# 8. 释放物块 + 张开夹爪
print("8. 释放物块...")
release_object()  # 先解除约束
time.sleep(0.3)
print("  张开夹爪...")
control_gripper(robot_id, 0.04)
time.sleep(0.5)

# 9. 提升机械臂回退
print("9. 提升机械臂...")
move_to_pose(robot_id, [target_pos[0], target_pos[1], target_pos[2] + 0.3], target_ori=down_ori)

print("\n任务完成！")
print("\n===== 夹子方向控制说明 =====")
print("1. 使用 calculateInverseKinematics 的 targetOrientation 参数设置姿态")
print("2. 四元数通过 getQuaternionFromEuler([roll, pitch, yaw]) 生成")
print("3. down_ori = [π, 0, 0]  → 夹子向下")
print("4. forward_ori = [π/2, 0, 0] → 夹子水平向前")
print("5. 可以调整欧拉角来改变夹子朝向")
time.sleep(3)
