#ifndef GLHEADERS_H_
#define GLHEADERS_H_

//#include <GL/glew.h>

#define GL3_PROTOTYPES
#include <GL/gl3.h>
#include <GL/glu.h>
#include <QDebug>
#include <QString>

/*
 * Append the CE() macro on the same line after an OpenGL function call to check for errors.
 * Use IF_FAIL("Error message") = ErrorCodeReturningFunction(); To print out a message on failure.
 */

class CheckSucc{
 public:
    CheckSucc(QString msg){
        succ = true;
        msg_ = msg;
    }

    CheckSucc operator = (bool rhs){
        succ = rhs;
        if(!rhs)
            qDebug() << msg_;
        return *this;
    }

    CheckSucc operator = (int rhs){
        if(rhs == -1){
            succ = false;
            qDebug() << msg_;
        }
        return *this;
    }

    bool operator == (bool rhs){
        return this->succ == rhs;
    }


 private:
    bool succ;
    QString msg_;
};

#define IF_FAIL(msg) CheckSucc(QString(QString(msg) + QString(" @ Line ") + QString::number(__LINE__) +  QString(" of ") +  QString(__FILE__)))

#define CC_GL_DEBUG 1

#ifdef CC_GL_DEBUG_
    static GLenum gl_err = 0;
    #define CE() gl_err = glGetError();\
    if (gl_err != GL_NO_ERROR) {\
        const char* err_str = reinterpret_cast<const char *>(gluErrorString(gl_err));\
        QString errString(err_str);\
        qDebug() << "GL Error:" << errString << "on line" << __LINE__ << "of" << __FILE__;\
    }
#else
    #define CE()
#endif  // CC_GL_DEBUG_

#endif  // GLHEADERS_H_
