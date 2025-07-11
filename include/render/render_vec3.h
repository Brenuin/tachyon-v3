#include "core/vec3.h"
#include <GLFW/glfw3.h>

namespace tachyon{

class RenderVec3 {
    public:
        RenderVec3(int width, int height, const char* title);
        ~RenderVec3();
    
        bool shouldClose() const;
        void beginFrame();
        void endFrame();
        void drawAxis(float length = 1.0f);
        void drawVector(const Vec3& origin, const Vec3& direction, const Vec3& color);
        void drawPoint(const Vec3& position, const Vec3& color, float size = 4.0f);
    
    private:
        GLFWwindow* window;
        int width, height;
    };
}