#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QImage>
#include <QPixmap>
#include <QFileInfo>
#include <QMessageBox>
#include "qtnn.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_openAction_clicked();

    void on_changeAction_clicked();

    void on_encryptAction_clicked();

    void on_decryptAction_clicked();
private:
    Ui::MainWindow *ui;
    int * m_image;      //上传的图片
//    struct LweSample enc_m_image[255];  //加密后的图片 它是STruct ，密文成员 直接enc_m_image[i].b
//    struct TFheGateBootstrappingSecretKeySet * m_secret; //生成的私钥
//    struct LweSample  enc_m_scores[255];
    struct LweSample * enc_m_image;  //加密后的图片 它是STruct ，密文成员 直接enc_m_image[i].b
    struct TFheGateBootstrappingSecretKeySet * m_secret; //生成的私钥
    struct LweSample * enc_m_scores;
    const struct LweParams * m_paras;

};
#endif // MAINWINDOW_H
