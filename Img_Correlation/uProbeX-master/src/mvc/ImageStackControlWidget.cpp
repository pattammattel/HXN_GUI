/*-----------------------------------------------------------------------------
 * Copyright (c) 2012, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#include <mvc/ImageStackControlWidget.h>


#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QIcon>
#include <QSplitter>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ImageStackControlWidget::ImageStackControlWidget(QWidget* parent) : QWidget(parent)
{
	createLayout();

}

/*---------------------------------------------------------------------------*/

ImageStackControlWidget::~ImageStackControlWidget()
{
	/* handeled by MapsWorkspaceMdodel
	for(auto &itr : _h5_model_map)
	{
		_model->unload_H5_Model(itr.first);
	}
	_h5_model_map.clear();
	*/
	if (_mapsFilsWidget != nullptr)
	{
		delete _mapsFilsWidget;
		_mapsFilsWidget = nullptr;
	}

}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::createLayout()
{
    QVBoxLayout* vlayout = new QVBoxLayout();
    QHBoxLayout* hlayout1 = new QHBoxLayout();
	QHBoxLayout* hlayout2 = new QHBoxLayout();
	_imageGrid = new MapsElementsWidget(1,1);
	_sws_widget = new SWSWidget();
	_mda_widget = new MDA_Widget();

	_mapsFilsWidget = new MapsWorkspaceFilesWidget();
	connect(_mapsFilsWidget, &MapsWorkspaceFilesWidget::loaded_model, this, &ImageStackControlWidget::onLoad_Model);
	connect(_mapsFilsWidget, &MapsWorkspaceFilesWidget::unloadList_model, this, &ImageStackControlWidget::onUnloadList_Model);

    _left_btn =  new QPushButton();
    _left_btn->setIcon(QIcon(":/images/previous.png"));
    _left_btn->setMaximumWidth(150);
	connect(_left_btn, &QAbstractButton::pressed, this, &ImageStackControlWidget::onPrevFilePressed);

    _image_name_cb = new QComboBox();
	_image_name_cb->setModel(&_image_name_cb_model);

    _right_btn =  new QPushButton();
    _right_btn->setIcon(QIcon(":/images/next.png"));
    _right_btn->setMaximumWidth(150);
	connect(_right_btn, &QAbstractButton::pressed, this, &ImageStackControlWidget::onNextFilePressed);

    hlayout1->addWidget(_left_btn);
    hlayout1->addWidget(_image_name_cb);
    hlayout1->addWidget(_right_btn);

	connect(_image_name_cb, SIGNAL(currentIndexChanged(QString)), this, SLOT(model_IndexChanged(QString)));

	connect(_mapsFilsWidget, &MapsWorkspaceFilesWidget::loaded_perc, this, &ImageStackControlWidget::update_progress_bar);

    vlayout->addItem(hlayout1);
	vlayout->addWidget(_imageGrid);
	vlayout->addWidget(_sws_widget);
	vlayout->addWidget(_mda_widget);

	hlayout2->addWidget(_mapsFilsWidget);

	QWidget *leftWidget = new QWidget();
	leftWidget->setLayout(hlayout2);
	QWidget *rightWidget = new QWidget();
	rightWidget->setLayout(vlayout);

	QSplitter* splitter = new QSplitter();
	splitter->setOrientation(Qt::Horizontal);
	splitter->addWidget(leftWidget);
	splitter->setStretchFactor(0, 1);
	splitter->addWidget(rightWidget);
	//createToolBar(m_imageViewWidget);
	//counts_layout->addWidget(m_toolbar);
	//counts_layout->addWidget(splitter);

	_load_progress = new QProgressBar();
	

	QVBoxLayout *mainLayout = new QVBoxLayout();
	mainLayout->addWidget(splitter);
	mainLayout->addWidget(_load_progress);
	

	//_imageGrid->hide();
	_sws_widget->hide();
	_mda_widget->hide();

	setLayout(mainLayout);
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::closeEvent(QCloseEvent *event)
{
	emit widgetClosed();
	event->accept();
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::model_IndexChanged(const QString &text)
{
	if (_h5_model_map.count(text) > 0)
	{
		if (_mda_widget->isVisible())
			_mda_widget->hide();
		if (_sws_widget->isVisible())
			_sws_widget->hide();
		_imageGrid->setModel(_h5_model_map[text]);
		_imageGrid->show();
	}
	else if(_raw_model_map.count(text) > 0)
	{
		if (_sws_widget->isVisible())
			_sws_widget->hide();
		if(_imageGrid->isVisible())
			_imageGrid->hide();
		_mda_widget->setModel(_raw_model_map[text]);
		_mda_widget->show();
	}
	else if (_sws_model_map.count(text) > 0)
	{
		if (_mda_widget->isVisible())
			_mda_widget->hide();
		if (_imageGrid->isVisible())
			_imageGrid->hide();
		_sws_widget->setModel(_sws_model_map[text]);
		_sws_widget->show();
	}
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::onPrevFilePressed()
{
	int idx = _image_name_cb->currentIndex();
	idx--;
	int amt = _image_name_cb->count();
	if (idx < amt)
	{
		_image_name_cb->setCurrentIndex(idx);
	}
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::onNextFilePressed()
{
	int idx = _image_name_cb->currentIndex();
	idx++;
	int amt = _image_name_cb->count();
	if (idx < amt)
	{
		_image_name_cb->setCurrentIndex(idx);
	}
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::setModel(MapsWorkspaceModel *model)
{
	_model = model;
	_mapsFilsWidget->clearLists();
	_mapsFilsWidget->setModel(_model);
	if (_model != nullptr)
	{
		_mapsFilsWidget->setLabelWorkspacePath(_model->get_directory_name());
		this->setWindowTitle(_model->get_directory_name());
	}
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::update_progress_bar(int val, int amt)
{
	if (amt != _load_progress->maximum())
	{
		_load_progress->setRange(0, amt);
	}

	_load_progress->setValue(val);
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::onLoad_Model(const QString name, MODEL_TYPE mt)
{
	QStringList model_list = _image_name_cb_model.stringList();

	//make sure no duplicates
	if (model_list.contains(name) == false)
	{
		switch (mt)
		{
		case MODEL_TYPE::MAPS_H5:
			_h5_model_map[name] = _model->get_MapsH5_Model(name);
			break;
		case MODEL_TYPE::RAW:
			_raw_model_map[name] = _model->get_RAW_Model(name);
			break;
		case MODEL_TYPE::SWS:
			_sws_model_map[name] = _model->get_SWS_Model(name);
			break;
		}
		
		_image_name_cb->addItem(name);
	}
}

/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::onUnloadList_Model(const QStringList sl, MODEL_TYPE mt)
{
	disconnect(_image_name_cb, SIGNAL(currentIndexChanged(QString)), this, SLOT(model_IndexChanged(QString)));

    _imageGrid->setModel(nullptr);
    foreach (QString s, sl)
    {
		switch (mt)
		{
		case MODEL_TYPE::MAPS_H5:
			_h5_model_map.erase(s);
			_model->unload_H5_Model(s);
			break;
		case MODEL_TYPE::RAW:
			_raw_model_map.erase(s);
			_model->unload_RAW_Model(s);
			break;
		case MODEL_TYPE::SWS:
			_sws_model_map.erase(s);
			_model->unload_SWS_Model(s);
			break;
		}
       
        for(int i=0; i < _image_name_cb->count(); i++)
        {
            if (s == _image_name_cb->itemText(i))
            {
                _image_name_cb->removeItem(i);
                break;
            }
        }
    }
	connect(_image_name_cb, SIGNAL(currentIndexChanged(QString)), this, SLOT(model_IndexChanged(QString)));
	emit _image_name_cb->currentIndexChanged(_image_name_cb->currentText());
}


/*---------------------------------------------------------------------------*/

void ImageStackControlWidget::update_file_list()
{
	if (_mapsFilsWidget != nullptr) 
	{
		_mapsFilsWidget->updatedMDA(); 
		_mapsFilsWidget->updatedHDF();
		_mapsFilsWidget->updatedVLM();
	} 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
