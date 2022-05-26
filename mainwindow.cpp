#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QPalette>
#include <QString>
#include "qtnn.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowIcon(QIcon("../untitled/Image/icon.jpg"));
    this->setWindowTitle(("李鹏博"));
    //设置按钮颜色
//    QPalette pal = (ui->openAction)->palette();
//    pal.setColor(QPalette::ButtonText, Qt::red);
//    al.setColor(QPalette::Button, Qt::green);
//    ui->openAction->setPalette(pal);
    
    

    this->m_image=new int[256];
    TFheGateBootstrappingParameterSet *params = our_default_gate_bootstrapping_parameters(80);
    this->m_paras=params->in_out_params;
    this->m_secret = new_random_gate_bootstrapping_secret_keyset(params);
    this->enc_m_image = new_LweSample_array(256, this->m_paras);
    this->enc_m_scores = new_LweSample_array(10 ,this->m_paras);
}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::on_openAction_clicked()
{
    //读取图片的路径
    QString filename;
    filename = QFileDialog::getOpenFileName(this, tr("Select image:"),
        "D:\\Documents\\Pictures", tr("Images (*.png *.bmp *.jpg *.gif)"));
    if (filename.isEmpty()) {
        return ;
    }
    //读取的图片
    QImage image;
    if (!image.load(filename)) {
        QMessageBox::information(this, tr("Error"), tr("Open file error"));
        return ;
    }

    setWindowTitle(QFileInfo(filename).fileName() + tr(" - imageViewer"));
    
    //转换灰度
    QImage imagegry = image.convertToFormat(QImage::Format_Grayscale8);
    
    //放大图片
    qreal width = pixmap.width();
    qreal height = pixmap.height();
    pixmap = pixmap.scaled(width*4,height*4,Qt::KeepAspectRatio);


    QPixmap pixmap = QPixmap::fromImage(imagegry); //
    qDebug() << "filname: " << pixmap;

    ui->imageLabel->setPixmap(pixmap);
    ui->imageLabel->resize(pixmap.size());

    ui->imageLabel->setPixmap(pixmap.scaled(ui->imageLabel->size(), Qt::IgnoreAspectRatio     , Qt::SmoothTransformation));

    unsigned char *pData=imagegry.bits();
    for(int i=0;i<16*16;i+=1)
    {
        if (pData[i]<128){
            pData[i]=0;
            m_image[i]=-1;
        }
        else{
            pData[i]=1;
            m_image[i]=1;
        }
        if (i%16 == 0)
            cout<<endl;
        cout<<int((m_image[i]))<<" ";
    }

}

void MainWindow::on_changeAction_clicked()
{
    //int image_class;
    net(this->enc_m_image,this->enc_m_scores,this->m_secret->cloud.bkFFT,this->m_paras);
    //ui->dataEdit->setText(QString::number(image_class));

}

void MainWindow::on_encryptAction_clicked()
{
    encrypt(this->m_image,this->enc_m_image,this->m_secret,this->m_paras);
    QImage imagegry_enc =  QImage(16, 16, QImage::Format_Grayscale8);
    int count =0;
    int grey=0;
    for(int x = 0; x<imagegry_enc.width(); x++){
        for(int y = 0; y<imagegry_enc.height(); y++){
            grey = int(this->enc_m_image[count].b/(2^32)*255);
            imagegry_enc.setPixel(x,y,grey);
            count+=1;
        }  
    }  

 
    QPixmap pixmap = QPixmap::fromImage(imagegry_enc); //
    qDebug() << "filname: " << pixmap;
    
    //放大图片
    qreal width = pixmap.width();
    qreal height = pixmap.height();
    pixmap = pixmap.scaled(width*4,height*4,Qt::KeepAspectRatio);

    //显示图片
    ui->imageLabel_2->setPixmap(pixmap);
    ui->imageLabel_2->resize(pixmap.size());
    ui->imageLabel_2->setPixmap(pixmap.scaled(ui->imageLabel_2->size(), Qt::IgnoreAspectRatio     , Qt::SmoothTransformation));



}
void MainWindow::on_decryptAction_clicked()
{
    int image_class;
    image_class=decrypt(this->enc_m_scores,this->m_secret);
    ui->dataEdit->setText(QString::number(image_class));

}

