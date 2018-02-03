#include <Bullet3Common/b3Quaternion.h>
#include <Bullet3OpenCL/BroadphaseCollision/b3GpuSapBroadphase.h>
#include <Bullet3OpenCL/BroadphaseCollision/b3GpuGridBroadphase.h>
#include <Bullet3OpenCL/Initialize/b3OpenCLUtils.h>

#include <Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h>
#include <Bullet3OpenCL/RigidBody/b3GpuRigidBodyPipeline.h>
#include <Bullet3OpenCL/RigidBody/b3GpuNarrowPhase.h>
#include <Bullet3Collision/NarrowPhaseCollision/b3Config.h>

#include <Bullet3Collision/BroadPhaseCollision/b3DynamicBvhBroadphase.h>
#include <Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h>
#include <Bullet3OpenCL/RigidBody/b3GpuNarrowPhaseInternalData.h>

#include <SFML/Graphics.hpp>

struct GpuDemoInternalData
{
	cl_platform_id m_platformId;
	cl_context m_clContext;
	cl_device_id m_clDevice;
	cl_command_queue m_clQueue;
	bool m_clInitialized;
	char*	m_clDeviceName;

	GpuDemoInternalData()
	:m_platformId(0),
	m_clContext(0),
	m_clDevice(0),
	m_clQueue(0),
	m_clInitialized(false),
	m_clDeviceName(0)
	{

	}
};

struct opencl_base
{
    struct GpuDemoInternalData*	m_clData;

    void init()
    {
        m_clData = new GpuDemoInternalData();

        cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

        int ciErrNum = 0;

        {
            m_clData->m_clContext = b3OpenCLUtils::createContextFromType(deviceType, &ciErrNum, 0,0,-1, -1,&m_clData->m_platformId);
        }


        oclCHECKERROR(ciErrNum, CL_SUCCESS);

        int numDev = b3OpenCLUtils::getNumDevices(m_clData->m_clContext);

        if (numDev>0)
        {
            m_clData->m_clDevice= b3OpenCLUtils::getDevice(m_clData->m_clContext,0);
            m_clData->m_clQueue = clCreateCommandQueue(m_clData->m_clContext, m_clData->m_clDevice, 0, &ciErrNum);
            oclCHECKERROR(ciErrNum, CL_SUCCESS);


            b3OpenCLDeviceInfo info;
            b3OpenCLUtils::getDeviceInfo(m_clData->m_clDevice,&info);
            m_clData->m_clDeviceName = info.m_deviceName;
            m_clData->m_clInitialized = true;

        }
    }
};

int main()
{
    sf::RenderWindow win(sf::VideoMode(800, 600), "openclfun");

    opencl_base base;
    base.init();

}
