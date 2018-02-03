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
#include <ocl/ocl.hpp>

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

#define MSTRINGIFY(A) #A

static const char* s_rigidBodyKernelString = MSTRINGIFY(

typedef struct
{
	float4 m_pos;
	float4 m_quat;
	float4 m_linVel;
	float4 m_angVel;
	unsigned int m_collidableIdx;
	float m_invMass;
	float m_restituitionCoeff;
	float m_frictionCoeff;
} Body;

__kernel void
	copyTransformsToVBOKernel( __global Body* gBodies, __global float4* posOrnColor, const int numNodes)
{
	int nodeID = get_global_id(0);
	if( nodeID < numNodes )
	{
		posOrnColor[nodeID] = (float4) (gBodies[nodeID].m_pos.xyz,1.0);
		posOrnColor[nodeID + numNodes] = gBodies[nodeID].m_quat;
	}
}
);

static const char* dname = "Phony device";

struct session_data
{
    cl_kernel m_copyTransformsToVBOKernel;

    b3OpenCLArray<b3Vector4>*	m_instancePosOrnColor;

	class b3GpuRigidBodyPipeline* m_rigidBodyPipeline;

	class b3GpuNarrowPhase* m_np;
	class b3GpuBroadphaseInterface* m_bp;
	class b3DynamicBvhBroadphase* m_broadphaseDbvt;

	b3Vector3 m_pickPivotInA;
	b3Vector3 m_pickPivotInB;
	float m_pickDistance;
	int m_pickBody;
	int	m_pickConstraint;

	int m_altPressed;
	int m_controlPressed;

	int m_pickFixedBody;
	int m_pickGraphicsShapeIndex;
	int m_pickGraphicsShapeInstance;
	b3Config m_config;

	session_data()
	{
	    m_instancePosOrnColor = nullptr;
	    m_rigidBodyPipeline = nullptr;

        m_copyTransformsToVBOKernel = 0;
		m_np = 0;
		m_bp = 0;
		m_broadphaseDbvt = 0;
		m_pickConstraint = -1;
		m_pickFixedBody = -1;
		m_pickGraphicsShapeIndex = -1;
		m_pickGraphicsShapeInstance = -1;
		m_pickBody = -1;
		m_altPressed = 0;
		m_controlPressed = 0;
	}
};

int gGpuArraySizeX = 45;
int gGpuArraySizeY = 55;
int gGpuArraySizeZ = 45;

extern int b3OpenCLUtils_clewInit();

struct opencl_base
{
    struct GpuDemoInternalData*	m_clData;

    session_data* m_data;

    void setupScene()
    {

    }

    void initCL()
    {
        m_clData = new GpuDemoInternalData();

        m_data = new session_data;

        b3OpenCLUtils_clewInit();

        cl::context ctx;
        cl::command_queue cqueue(ctx);

        m_clData->m_clContext = ctx.ccontext;
        m_clData->m_platformId = ctx.platform;
        m_clData->m_clDevice = ctx.selected_device;
        m_clData->m_clQueue = cqueue.cqueue;

        m_clData->m_clInitialized = true;
        m_clData->m_clDeviceName = "PhonyDevice";

        int errNum = 0;

        cl::program prog(ctx, s_rigidBodyKernelString, false);
        prog.build_with(ctx, "");

        cl::kernel copyTransformsToVBOKernel(prog, "copyTransformsToVBOKernel");

        m_data->m_copyTransformsToVBOKernel = copyTransformsToVBOKernel.ckernel;


        m_data->m_config.m_maxConvexBodies = b3Max(m_data->m_config.m_maxConvexBodies,gGpuArraySizeX*gGpuArraySizeY*gGpuArraySizeZ+10);
		m_data->m_config.m_maxConvexShapes = m_data->m_config.m_maxConvexBodies;
		int maxPairsPerBody = 16;
		m_data->m_config.m_maxBroadphasePairs = maxPairsPerBody*m_data->m_config.m_maxConvexBodies;
		m_data->m_config.m_maxContactCapacity = m_data->m_config.m_maxBroadphasePairs;

        b3GpuNarrowPhase* np = new b3GpuNarrowPhase(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue,m_data->m_config);
		b3GpuBroadphaseInterface* bp =0;


		bool useUniformGrid = false;

		if (useUniformGrid)
		{
			bp = new b3GpuGridBroadphase(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue);
		} else
		{
			bp = new b3GpuSapBroadphase(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue);
		}

		m_data->m_np = np;
		m_data->m_bp = bp;
		m_data->m_broadphaseDbvt = new b3DynamicBvhBroadphase(m_data->m_config.m_maxConvexBodies);

		m_data->m_rigidBodyPipeline = new b3GpuRigidBodyPipeline(m_clData->m_clContext,m_clData->m_clDevice,m_clData->m_clQueue, np, bp,m_data->m_broadphaseDbvt,m_data->m_config);

		b3Vector3 gravity = b3MakeVector3(0, -9.8, 0);

        m_data->m_rigidBodyPipeline->setGravity(gravity);


		setupScene();

		m_data->m_rigidBodyPipeline->writeAllInstancesToGpu();
		np->writeAllBodiesToGpu();
		bp->writeAabbsToGpu();


        int index = 0;


        {
            float rad = 5.f;
            float mass = 1.f;

            int colIndex = m_data->m_np->registerSphereShape(rad);

            b3Vector3 position = b3MakeVector3(0,0,0);

            b3Quaternion orn(0,0,0,1);

            b3Vector4 scaling = b3MakeVector4(rad, rad, rad, 1.f);

            int pid = m_data->m_rigidBodyPipeline->registerPhysicsInstance(mass, position, orn, colIndex, index, false);

            index++;
        }


        m_data->m_rigidBodyPipeline->writeAllInstancesToGpu();
		np->writeAllBodiesToGpu();
		bp->writeAabbsToGpu();
    }

    void tick(double timestep_s)
    {
        int num_objects = m_data->m_rigidBodyPipeline->getNumBodies();

        {
            B3_PROFILE("stepSimulation");
            m_data->m_rigidBodyPipeline->stepSimulation(1./60.f);
        }

        bool convertOnCpu = false;

        /*if(num_objects)
        {
            if (convertOnCpu)
            {
                b3GpuNarrowPhaseInternalData*	npData = m_data->m_np->getInternalData();
                npData->m_bodyBufferGPU->copyToHost(*npData->m_bodyBufferCPU);

                b3AlignedObjectArray<b3Vector4> vboCPU;
                m_data->m_instancePosOrnColor->copyToHost(vboCPU);

                for (int i=0;i<num_objects;i++)
                {
                    b3Vector4 pos = (const b3Vector4&)npData->m_bodyBufferCPU->at(i).m_pos;
                    b3Quat orn = npData->m_bodyBufferCPU->at(i).m_quat;
                    pos.w = 1.f;
                    vboCPU[i] = pos;
                    vboCPU[i + num_objects] = (b3Vector4&)orn;
                }
                m_data->m_instancePosOrnColor->copyFromHost(vboCPU);

            } else
            {
                B3_PROFILE("cl2gl_convert");
                int ciErrNum = 0;
                cl_mem bodies = m_data->m_rigidBodyPipeline->getBodyBuffer();
                b3LauncherCL launch(m_clData->m_clQueue,m_data->m_copyTransformsToVBOKernel,"m_copyTransformsToVBOKernel");
                launch.setBuffer(bodies);
                launch.setBuffer(m_data->m_instancePosOrnColor->getBufferCL());
                launch.setConst(num_objects);
                launch.launch1D(num_objects);
                oclCHECKERROR(ciErrNum, CL_SUCCESS);
            }
        }*/

        printf("%i nobj\n", num_objects);

        if(num_objects)
        {
            b3GpuNarrowPhaseInternalData*	npData = m_data->m_np->getInternalData();
            npData->m_bodyBufferGPU->copyToHost(*npData->m_bodyBufferCPU);

            for(int i=0; i < num_objects; i++)
            {
                b3Vector4 pos = (const b3Vector4&)npData->m_bodyBufferCPU->at(i).m_pos;

                printf("%f %f %f\n", pos.x, pos.y, pos.z);
            }

            /*b3AlignedObjectArray<b3Vector4> vboCPU;
            m_data->m_instancePosOrnColor->copyToHost(vboCPU);

            for (int i=0;i<num_objects;i++)
            {
                b3Vector4 pos = (const b3Vector4&)npData->m_bodyBufferCPU->at(i).m_pos;
                b3Quat orn = npData->m_bodyBufferCPU->at(i).m_quat;
                pos.w = 1.f;
                vboCPU[i] = pos;
                vboCPU[i + num_objects] = (b3Vector4&)orn;
            }
            m_data->m_instancePosOrnColor->copyFromHost(vboCPU);*/



        }
    }
};

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    sf::RenderWindow win(sf::VideoMode(800, 600), "openclfun");

    opencl_base base;
    base.initCL();

    sf::Clock clk;
    sf::Keyboard key;

    while(win.isOpen())
    {
        sf::Event event;

        while(win.pollEvent(event))
        {
            if(event.type == sf::Event::Closed)
                win.close();
        }

        double timestep_s = clk.restart().asMicroseconds() / 1000. / 1000.;

        base.tick(timestep_s);

        if(key.isKeyPressed(sf::Keyboard::N))
        {
            std::cout << timestep_s << std::endl;
        }

        win.display();
        win.clear();
    }
}
