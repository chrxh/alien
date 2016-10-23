#include "computercodeedit.h"

#include "global/editorsettings.h"
#include "global/guisettings.h"

#include <QKeyEvent>
#include <QTextBlock>
#include <QScrollBar>

ComputerCodeEdit::ComputerCodeEdit(QWidget *parent)
    : QTextEdit(parent)
{
    QTextEdit::setTextInteractionFlags(Qt::TextSelectableByKeyboard | Qt::TextEditable);
    verticalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
}

ComputerCodeEdit::~ComputerCodeEdit()
{

}

void ComputerCodeEdit::update (QString code)
{
    displayData(code);
}

void ComputerCodeEdit::update ()
{
    QTextEdit::setText("");
}

QString ComputerCodeEdit::getCode ()
{
    removeLineNumbers();
    QString code = QTextEdit::toPlainText();
    insertLineNumbers();
    return code;
}

void ComputerCodeEdit::keyPressEvent (QKeyEvent* e)
{
    removeLineNumbers();
    QTextEdit::keyPressEvent(e);
    insertLineNumbers();
}

void ComputerCodeEdit::mousePressEvent(QMouseEvent* e)
{
    QTextEdit::mousePressEvent(e);
    int pos(QTextEdit::textCursor().positionInBlock());
    if( pos < 3 )
        for( int i = 0; i < 3-pos; i++ )
            QTextEdit::moveCursor(QTextCursor::Right);
}

void ComputerCodeEdit::wheelEvent (QWheelEvent* e)
{
    QTextEdit::wheelEvent(e);
    QTextEdit::clearFocus();
}


void ComputerCodeEdit::displayData (QString code)
{
    //set colors
    QPalette p(QTextEdit::palette());
    p.setColor(QPalette::Text, CELL_EDIT_DATA_COLOR1);
    QTextEdit::setPalette(p);

    //replace < and > by HTML characters
    code.replace("<", "&lt;");
    code.replace(">", "&gt;");

    //define auxilliary strings
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorDataStart2 = "<span style=\"color:"+CELL_EDIT_DATA_COLOR2.name()+"\">";
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorEnd = "</span>";
    QString text;

    code.replace(" ", "&nbsp;");
    int eol = 0;
    int line = 1;
    do {
        eol = code.indexOf("\n");
        text += parStart + colorDataStart2;
        QString lineNumber = QString("%1").arg(line, 2);
        lineNumber.replace(" ", "0");
        text += lineNumber;
        text += colorEnd + colorDataStart;
        text += " " + code.left(eol);
        text += colorEnd + parEnd;
        code.remove(0,eol+1);
        line++;
    } while(eol != -1);

    QTextEdit::setText(text);
}

void ComputerCodeEdit::insertLineNumbers ()
{
    //define auxilliary strings
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorDataStart2 = "<span style=\"color:"+CELL_EDIT_DATA_COLOR2.name()+"\">";
    QString colorEnd = "</span>";

    //insert line numbers
    QTextCursor c(QTextEdit::document());
    int line = 1;
    do {
        QString text;
        QString lineNumber = QString("%1").arg(line, 2);
        lineNumber.replace(" ", "0");
        text += colorDataStart2;
        text += lineNumber;
        text += colorEnd + colorDataStart;
        text += " " + colorEnd;
        c.insertHtml(text);
        line++;
    }
    while(c.movePosition(QTextCursor::NextBlock));
}

void ComputerCodeEdit::removeLineNumbers ()
{
    //remove line numbers
    QTextCursor c(QTextEdit::document());
    do {
        c.deleteChar();
        c.deleteChar();
        c.deleteChar();
    }
    while(c.movePosition(QTextCursor::NextBlock));

}


