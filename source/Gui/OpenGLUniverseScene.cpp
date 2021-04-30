#include "OpenGLUniverseScene.h"

#include "EngineInterface/SimulationAccess.h"

float vertices[] = {
    // positions          // colors           // texture coords
    1.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,  // top right
    1.0f,  -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,  // bottom right
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,  // bottom left
    -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f   // top left
};
unsigned int indices[] = {
    // note that we start from 0!
    0,
    1,
    3,  // first Triangle
    1,
    2,
    3  // second Triangle
};

#include <iostream>


OpenGLUniverseScene::OpenGLUniverseScene(
    SimulationAccess* access,
    IntVector2D const& viewSize,
    std::mutex& mutex,
    QObject* parent /*= nullptr*/)
    : QGraphicsScene(parent)
    , _access(access)
    , _mutex(mutex)
{
    setSceneRect(0, 0, viewSize.x - 3, viewSize.y - 3);

    m_context = new QOpenGLContext(parent);
    m_context->create();
    initializeOpenGLFunctions();

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glGenTextures(1, &texture);
    updateTexture();

    glBindTexture(GL_TEXTURE_2D, texture);
    _imageResource = _access->registerImageResource(texture);
    glBindTexture(GL_TEXTURE_2D, 0);

    //        _image = boost::make_shared<QImage>(_displaySize.x, _displaySize.y, QImage::Format_ARGB32);
}

ImageResource OpenGLUniverseScene::getImageResource() const
{
    return _imageResource;
}

void OpenGLUniverseScene::updateTexture()
{
    glBindTexture(GL_TEXTURE_2D, texture);
    // set the texture wrapping parameters
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_S,
        GL_REPEAT);  // set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // load image, create texture and generate mipmaps

    QImage image(
        static_cast<int>(sceneRect().width() + 0.5f),
        static_cast<int>(sceneRect().height() + 0.5f),
        QImage::Format_ARGB32);

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, sceneRect().width(), sceneRect().height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);


    /*
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA8UI_EXT,
            1000,
            1000,
            0,
            GL_RGBA_INTEGER_EXT,
            GL_UNSIGNED_BYTE,
            NULL);
*/


    glGenerateMipmap(GL_TEXTURE_2D);

    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLUniverseScene::drawBackground(QPainter* painter, const QRectF& rect)
{
    std::lock_guard<std::mutex> lock(_mutex);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0);
    //--------
    /*
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
*/

    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    // draw our first triangle
    glUseProgram(shaderProgram);

    glBindVertexArray(VAO);
    //glDrawArrays(GL_TRIANGLES, 0, 6);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    //        glDeleteTextures(1, &texture1);
}
