import QtQuick

Rectangle {
    id: root
    property int currentIndex: 0
    signal tabChanged(int index)

    height: 64
    color: Qt.rgba(0.02, 0.06, 0.18, 0.92)

    Rectangle {
        anchors { top: parent.top; left: parent.left; right: parent.right }
        height: 1
        color: Qt.rgba(1, 1, 1, 0.20)
    }

    Row {
        anchors.fill: parent

        Repeater {
            model: [
                { icon: "⌂",  label: "Главная"   },
                { icon: "≡",  label: "Операции"  },
                { icon: "◎",  label: "Бюджет"    },
                { icon: "⊞",  label: "Отчёты"    },
                { icon: "⚙",  label: "Семья"     }
            ]

            delegate: Item {
                width: parent.width / 5
                height: parent.height

                Rectangle {
                    anchors { fill: parent; margins: 4 }
                    radius: 12
                    color: root.currentIndex === index
                           ? Qt.rgba(0.3, 0.65, 1.0, 0.18) : "transparent"
                    Behavior on color { ColorAnimation { duration: 150 } }

                    Column {
                        anchors.centerIn: parent
                        spacing: 2

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: modelData.icon
                            font.pixelSize: 22
                            color: root.currentIndex === index ? "#4FC3F7" : Qt.rgba(1,1,1,0.48)
                            Behavior on color { ColorAnimation { duration: 150 } }
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: modelData.label
                            font.pixelSize: 10
                            color: root.currentIndex === index ? "#4FC3F7" : Qt.rgba(1,1,1,0.48)
                            Behavior on color { ColorAnimation { duration: 150 } }
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: root.tabChanged(index)
                    }
                }
            }
        }
    }
}
