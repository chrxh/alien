#include <QKeyEvent>
#include <QTextBlock>
#include <QTextLayout>
#include <qmath.h>

#include "Gui/Settings.h"

#include "ParticleEditTab.h"
#include "DataEditModel.h"
#include "DataEditController.h"

ParticleEditTab::ParticleEditTab(QWidget *parent) :
    QTextEdit(parent)
{
    setFont(GuiSettings::getGlobalFont());
    QTextEdit::setTextInteractionFlags(Qt::TextSelectableByKeyboard | Qt::TextEditable);
}

void ParticleEditTab::keyPressEvent (QKeyEvent* e)
{
    //auxilliary data
    QString colorDataStart = "<span style=\"color:"+Const::CellEditDataColor1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+Const::CellEditDataColor2.name()+"\">";
    QString colorEnd = "</span>";
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();
    int rowLen = QTextEdit::document()->findBlockByNumber(row).length();

    //request update?
	if ((e->key() == Qt::Key_Down) || (e->key() == Qt::Key_Up) || (e->key() == Qt::Key_Enter) || (e->key() == Qt::Key_Return)) {
		updateModelAndNotifyController();
	}

    //typing number?
    QString k;
    if( (e->key() == Qt::Key_0) )
        k = "0";
    if( (e->key() == Qt::Key_1) )
        k = "1";
    if( (e->key() == Qt::Key_2) )
        k = "2";
    if( (e->key() == Qt::Key_3) )
        k = "3";
    if( (e->key() == Qt::Key_4) )
        k = "4";
    if( (e->key() == Qt::Key_5) )
        k = "5";
    if( (e->key() == Qt::Key_6) )
        k = "6";
    if( (e->key() == Qt::Key_7) )
        k = "7";
    if( (e->key() == Qt::Key_8) )
        k = "8";
    if( (e->key() == Qt::Key_9) )
        k = "9";


    //check for end of line in the case typing number or period
    if( (!k.isEmpty()) || (e->key() == Qt::Key_Period) ) {
        if( QTextEdit::textCursor().block().text().length() > 36 )
            return;
    }

    //insert number
    if( !k.isEmpty() ) {
        int s = QTextEdit::textCursor().block().text().indexOf(".");

        //no dot before cursor?
        if( (s < 0) || (s >= col) )
            QTextEdit::textCursor().insertHtml(colorDataStart+k+colorEnd);
        else
            QTextEdit::textCursor().insertHtml(colorData2Start+k+colorEnd);
    }

    //typing dot?
    if( e->key() == Qt::Key_Period ) {
        int s = QTextEdit::textCursor().block().text().indexOf(".");

        //there is a dot after cursor?
        if( (s < 0) || (s >= col) ) {
            QTextEdit::textCursor().insertHtml(colorData2Start+"."+colorEnd);

            //removes other dots and recolor the characters from new dot to line end
            int n = rowLen-col-1;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n).remove('.');//.left(s-col);
            for( int i = 0; i < n; ++i )
                QTextEdit::textCursor().deleteChar();
            QTextEdit::textCursor().insertHtml(colorData2Start+t+colorEnd);
            for( int i = 0; i < t.size(); ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }

        //there is a dot before cursor?
        if( (s >= 0) && (s < col) ) {

            //removes other dots and recolor the characters from old dot to line end
            int n = rowLen-s-1;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n).remove('.');//.left(s-col);
            QTextCursor c = QTextEdit::textCursor();
            c.movePosition(QTextCursor::Left,QTextCursor::MoveAnchor, col-s);
            for( int i = 0; i < n; ++i )
                c.deleteChar();
            c.insertHtml(colorDataStart+t.left(col-s-1)+colorEnd);
            c.insertHtml(colorData2Start+"."+t.right(n-col+s)+colorEnd);
            for( int i = 0; i < n-col+s; ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }
    }

    //typing minus
    if( e->key() == Qt::Key_Minus ) {
        if( row < 4 )
            QTextEdit::textCursor().insertHtml("-");
    }

    //typing backspace?
    if( (col > 24) && (e->key() == Qt::Key_Backspace) ) {

        //is there a dot at cursor's position?
        QChar c = QTextEdit::document()->characterAt(QTextEdit::textCursor().position()-1);
        if( c == '.' ) {
            QTextEdit::keyPressEvent(e);

            //recolor the characters from dot to line end
            int n = rowLen-col-1;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n);
            for( int i = 0; i < n; ++i )
                QTextEdit::textCursor().deleteChar();
            QTextEdit::textCursor().insertHtml(colorDataStart+t+colorEnd);
            for( int i = 0; i < n; ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }
        else
            QTextEdit::keyPressEvent(e);
    }

    //typing delete?
    if( (col < rowLen-1) && (e->key() == Qt::Key_Delete) ) {

        //is there a dot at cursor's position?
        QChar c = QTextEdit::document()->characterAt(QTextEdit::textCursor().position());
        if( c == '.' ) {
            QTextEdit::keyPressEvent(e);

            //recolor the characters from dot to line end
            int n = rowLen-col-2;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n);
            for( int i = 0; i < n; ++i )
                QTextEdit::textCursor().deleteChar();
            QTextEdit::textCursor().insertHtml(colorDataStart+t+colorEnd);
            for( int i = 0; i < n; ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }
        else
            QTextEdit::keyPressEvent(e);
    }

    QString currentText = QTextEdit::textCursor().block().text();

    //typing left button?
    if( (col > 24) && (e->key() == Qt::Key_Left) )
            QTextEdit::keyPressEvent(e);

    //typing right button?
    if( (col < rowLen-1) && (e->key() == Qt::Key_Right) )
            QTextEdit::keyPressEvent(e);

    //typing down button?
    if( e->key() == Qt::Key_Down )
        if( row < 4 )
            QTextEdit::keyPressEvent(e);

    //typing up button?
    if( e->key() == Qt::Key_Up )
        if( row > 0 )
            QTextEdit::keyPressEvent(e);
}

void ParticleEditTab::mousePressEvent(QMouseEvent* e)
{
    QTextEdit::mousePressEvent(e);
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();

    //move cursor to correct position
    if( col < 24 ) {
        QTextEdit::moveCursor(QTextCursor::StartOfBlock);
        QTextEdit::moveCursor(QTextCursor::NextWord);
        QTextEdit::moveCursor(QTextCursor::NextWord);
        QTextEdit::moveCursor(QTextCursor::NextWord);
    }
}

void ParticleEditTab::mouseDoubleClickEvent (QMouseEvent* e)
{
    QTextEdit::clearFocus();
}

void ParticleEditTab::wheelEvent (QWheelEvent* e)
{
    QTextEdit::wheelEvent(e);
    QTextEdit::clearFocus();
}

void ParticleEditTab::init(DataEditModel * model, DataEditController * controller)
{
	_model = model;
	_controller = controller;
}

void ParticleEditTab::updateDisplay ()
{
    auto particle = _model->getParticleToEditRef();
    if (!particle) {
        return;
    }

    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();

    //define auxiliary strings
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorTextStart = "<span style=\"color:"+Const::CellEditTextColor1.name()+"\">";
    QString colorEnd = "</span>";
    QString text;

    //set cursor color
    QPalette p(QTextEdit::palette());
    p.setColor(QPalette::Text, Const::CellEditCursorColor);
    QTextEdit::setPalette(p);

    //create string of display
	text = parStart + colorTextStart + "position x: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + colorEnd;
    text += generateFormattedRealString(particle->pos->x())+parEnd;
    text += parStart+colorTextStart+ "position y: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(particle->pos->y())+parEnd;
    text += parStart+colorTextStart+ "velocity x: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(particle->vel->x())+parEnd;
    text += parStart+colorTextStart+ "velocity y: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(particle->vel->y())+parEnd;
    text += parStart+colorTextStart+ "energy value: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(*particle->energy)+parEnd;
    QTextEdit::setText(text);

    //restore cursor
    for( int i = 0; i < row; ++i )
        QTextEdit::moveCursor(QTextCursor::NextBlock);
    for( int i = 0; i < col; ++i )
        QTextEdit::moveCursor(QTextCursor::Right);
}

void ParticleEditTab::updateModelAndNotifyController()
{
    auto particle = _model->getParticleToEditRef();
    if (!particle) {
        return;
    }

    int row = QTextEdit::textCursor().blockNumber();
	QString currentText = QTextEdit::textCursor().block().text();

	if (row == 0)
		particle->pos->setX(generateNumberFromFormattedString(currentText));
	if (row == 1)
		particle->pos->setY(generateNumberFromFormattedString(currentText));
	if (row == 2)
		particle->vel->setX(generateNumberFromFormattedString(currentText));
	if (row == 3)
		particle->vel->setY(generateNumberFromFormattedString(currentText));
	if (row == 4)
		particle->energy = generateNumberFromFormattedString(currentText);
	_controller->notificationFromParticleTab();
}

qreal ParticleEditTab::generateNumberFromFormattedString (QString s)
{
    int i = s.indexOf(':');
    if( i >= 0 ) {
        QString sub = s.right(s.size()-i-1);
        qreal d = sub.toDouble();
        if( d >= 0.0 )
            d += 0.00005;
        else
            d -= 0.00005;
        return d;
    }
    return 0.0;
}

QString ParticleEditTab::generateFormattedRealString (QString s)
{
    QString colorDataStart = "<span style=\"color:"+Const::CellEditDataColor1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+Const::CellEditDataColor2.name()+"\">";
    QString colorEnd = "</span>";
    QString iS, reS;
    int i = s.indexOf(".");
    if( i == -1 )
        iS = s;
    else {
        iS = s.left(i);
        reS = s.remove(0,i+1);
    }
    return colorDataStart+iS+colorEnd+colorData2Start+"."+reS+colorEnd;
}

QString ParticleEditTab::generateFormattedRealString (qreal r)
{
    QString colorDataStart = "<span style=\"color:"+Const::CellEditDataColor1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+Const::CellEditDataColor2.name()+"\">";
    QString colorEnd = "</span>";
    bool negativeSign = false;
    if( r < 0.0 ) {
        r = -r;
        negativeSign = true;
    }
    int i = qFloor(r);
    int re = (r-qFloor(r))*10000.0;
    QString iS = QString("%1").arg(i);
    QString reS = QString("%1").arg(re, 4);
    reS.replace(" ", "0");
    if( negativeSign)
        return colorDataStart+"-"+iS+colorEnd+colorData2Start+"."+reS+colorEnd;
    else
        return colorDataStart+iS+colorEnd+colorData2Start+"."+reS+colorEnd;
}

