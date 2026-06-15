#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QLocale>
#include <QTranslator>
#include "database/database.h"
#include "models/usermanager.h"
#include "models/budgetmanager.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    app.setOrganizationName("FamilyBudget");
    app.setOrganizationDomain("familybudget.app");
    app.setApplicationName("FamilyBudget");
    app.setApplicationVersion("1.0.0");

    QQuickStyle::setStyle("Basic");

    Database db;
    if (!db.initialize()) {
        return 1;
    }

    UserManager userManager(&db);
    BudgetManager budgetManager(&db);

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("userManager", &userManager);
    engine.rootContext()->setContextProperty("budgetManager", &budgetManager);

    const QUrl url(QStringLiteral("qrc:/qt/qml/FamilyBudget/qml/main.qml"));
    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreationFailed,
        &app, []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
