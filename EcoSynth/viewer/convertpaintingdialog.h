#ifndef CONVERTPAINTINGDIALOG_H
#define CONVERTPAINTINGDIALOG_H

#include <QDialog>

namespace Ui {
class ConvertPaintingDialog;
}

class ConvertPaintingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ConvertPaintingDialog(QWidget *parent = 0);

    void get_values(int &from, int &to);

    ~ConvertPaintingDialog();

private:
    Ui::ConvertPaintingDialog *ui;
};

#endif // CONVERTPAINTINGDIALOG_H
