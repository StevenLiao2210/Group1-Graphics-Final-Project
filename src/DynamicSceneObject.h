#pragma once

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "SceneManager.h"

class DynamicSceneObject 
{

private:
	GLuint m_indexBufferHandle;
	float *m_dataBuffer = nullptr;
	unsigned int *m_indexBuffer = nullptr;

	GLuint m_vao;
	GLuint m_dataBufferHandle;
	GLenum m_primitive;
	int m_pixelFunctionId;
	int m_indexCount;

	glm::mat4 m_modelMat;

	//per object albedo texture
	GLuint m_albedoTexHandle = 0;

	// normal mapping
	GLuint m_normalTex = 0;    
    bool   m_useNormalMapping = false; 


public:
	DynamicSceneObject(const int maxNumVertex, const int maxNumIndex, const bool normalFlag, const bool uvFlag);
	virtual ~DynamicSceneObject();

	void update();

	float* dataBuffer();
	unsigned int *indexBuffer();

	void updateDataBuffer(const int byteOffset, const int dataByte);
	void updateIndexBuffer(const int byteOffset, const int dataByte);

	void setPixelFunctionId(const int functionId);
	void setPrimitive(const GLenum primitive);
	void setModelMat(const glm::mat4& modelMat);

	void setAlbedoTexture(GLuint tex);
	// normal mapping
	void setNormalTexture(GLuint tex) { m_normalTex = tex; }
    void setUseNormalMapping(bool v) { m_useNormalMapping = v; }
	GLuint normalTexture() const      { return m_normalTex; }
};

