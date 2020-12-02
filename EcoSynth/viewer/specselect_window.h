#ifndef SPECSELECT_WINDOW
#define SPECSELECT_WINDOW


#include <QWidget>
#include <QCheckBox>

#include "data_importer/data_importer.h"

class Window;

class specselect_window : public QWidget
{
    Q_OBJECT
public:
    specselect_window(data_importer::common_data cdata, Window *parent);
    specselect_window(std::string dbname, Window *parent);

    void add_widget(QWidget *w);

private:
    std::map<int, QCheckBox *> cboxes;

private slots:
    void statechanged(int state);
public slots:
    void disable();
    void enable();

signals:
    void species_added(int id);
    void species_removed(int id);
};

#endif // SPECSELECT_WINDOW
