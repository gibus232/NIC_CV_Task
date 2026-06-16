import QtQuick
import QtQuick.Controls
import FamilyBudget

Item {
    id: root
    property string month: budgetManager.currentMonth()
    property double totalExpenses: 0
    property double income: 0
    property var catData: []

    function loadData() {
        var cats = budgetManager.getExpensesByCategory(userManager.currentUserId, month)
        catData = cats
        var total = 0
        for (var i = 0; i < cats.length; i++) total += cats[i].total
        totalExpenses = total
        income = budgetManager.getTotalIncome(userManager.currentUserId, month)
    }

    Component.onCompleted: loadData()
    Connections { target: budgetManager; function onDataChanged() { loadData() } }

    AeroHeader {
        id: hdr
        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
        title: "Отчёты"
    }

    Flickable {
        anchors.top: hdr.bottom; anchors.left: parent.left; anchors.right: parent.right
        anchors.bottom: parent.bottom
        contentHeight: contentCol.height + 24
        clip: true

        Column {
            id: contentCol
            width: parent.width
            spacing: 0

            // Month selector
            Item { width: parent.width; height: 12 }
            Row {
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 12

                Rectangle {
                    width: 36; height: 36; radius: 18
                    color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
                    Text { anchors.centerIn: parent; text: "‹"; color: "white"; font.pixelSize: 24 }
                    MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.prevMonth(root.month); root.loadData() } }
                }
                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    text: budgetManager.monthDisplayName(root.month)
                    color: "white"; font.pixelSize: 16; font.weight: Font.Medium
                }
                Rectangle {
                    width: 36; height: 36; radius: 18
                    color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
                    Text { anchors.centerIn: parent; text: "›"; color: "white"; font.pixelSize: 24 }
                    MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.nextMonth(root.month); root.loadData() } }
                }
            }
            Item { width: parent.width; height: 12 }

            // Income / Expense summary cards
            Row {
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 12

                Rectangle {
                    width: 148; height: 68; radius: 18
                    color: Qt.rgba(0.08,0.38,0.08,0.28)
                    border.color: Qt.rgba(0.35,0.85,0.35,0.28); border.width: 1
                    Column {
                        anchors.centerIn: parent; spacing: 5
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "↑ Доход"; color: "#A5D6A7"; font.pixelSize: 12 }
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: Number(root.income).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"; color: "#66BB6A"; font.pixelSize: 18; font.weight: Font.SemiBold }
                    }
                }
                Rectangle {
                    width: 148; height: 68; radius: 18
                    color: Qt.rgba(0.38,0.08,0.08,0.28)
                    border.color: Qt.rgba(0.85,0.30,0.30,0.28); border.width: 1
                    Column {
                        anchors.centerIn: parent; spacing: 5
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "↓ Расход"; color: "#EF9A9A"; font.pixelSize: 12 }
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: Number(root.totalExpenses).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"; color: "#EF5350"; font.pixelSize: 18; font.weight: Font.SemiBold }
                    }
                }
            }
            Item { width: parent.width; height: 20 }

            // Section label
            Text {
                anchors.left: parent.left; anchors.leftMargin: 20
                text: "Расходы по категориям"
                color: Qt.rgba(1,1,1,0.50); font.pixelSize: 12
            }
            Item { width: parent.width; height: 10 }

            // Category bars
            Column {
                width: parent.width - 32
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 10

                Repeater {
                    model: root.catData
                    delegate: Rectangle {
                        id: bar
                        width: parent.width; height: 60; radius: 14
                        color: Qt.rgba(1,1,1,0.07); border.color: Qt.rgba(1,1,1,0.14); border.width: 1

                        property double pct: root.totalExpenses > 0 ? modelData.total / root.totalExpenses : 0

                        // Glossy highlight
                        Rectangle {
                            anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                            height: parent.height * 0.45; radius: parent.radius
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.08) }
                                GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
                            }
                        }

                        Column {
                            anchors.left: parent.left; anchors.right: parent.right
                            anchors.leftMargin: 14; anchors.rightMargin: 14
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: 7

                            Item {
                                width: parent.width; height: 16
                                Rectangle {
                                    width: 10; height: 10; radius: 5
                                    anchors.verticalCenter: parent.verticalCenter
                                    color: modelData.color || "#1565C0"
                                }
                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.left: parent.left; anchors.leftMargin: 18
                                    anchors.right: amtLabel.left; anchors.rightMargin: 8
                                    text: modelData.name
                                    color: "white"; font.pixelSize: 13
                                    elide: Text.ElideRight
                                }
                                Text {
                                    id: amtLabel
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.right: parent.right
                                    text: Number(modelData.total).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"
                                    color: Qt.rgba(1,1,1,0.75); font.pixelSize: 13
                                }
                            }

                            // Progress bar
                            Rectangle {
                                width: parent.width; height: 5; radius: 3
                                color: Qt.rgba(1,1,1,0.10)
                                Rectangle {
                                    width: parent.width * bar.pct
                                    height: parent.height; radius: parent.radius
                                    color: modelData.color || "#1565C0"
                                    Behavior on width { NumberAnimation { duration: 400; easing.type: Easing.OutCubic } }
                                }
                            }
                        }
                    }
                }

                // Empty state
                Rectangle {
                    visible: root.catData.length === 0
                    width: parent.width; height: 80; radius: 16
                    color: Qt.rgba(1,1,1,0.05); border.color: Qt.rgba(1,1,1,0.12); border.width: 1
                    Text {
                        anchors.centerIn: parent
                        text: "Расходов за этот месяц нет"
                        color: Qt.rgba(1,1,1,0.40); font.pixelSize: 15
                    }
                }
            }

            Item { width: parent.width; height: 16 }
        }
    }
}
