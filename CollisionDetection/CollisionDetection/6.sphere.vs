#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;
// vec3 bPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float radius;

void main()
{
    // bPos = aPos * radius;
    FragPos = vec3(model * vec4(aPos * radius, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal; 
    gl_Position = projection * view * vec4(FragPos, 1.0);
}