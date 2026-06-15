import QtQuick

Item {
    id: root
    property string pin: ""
    property int maxLength: 4
    signal pinComplete(string pin)

    width: 280; height: 300

    Column {
        anchors.fill: parent
        spacing: 24

        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 18
            Repeater {
                model: root.maxLength
                delegate: Rectangle {
                    width: 16; height: 16; radius: 8
                    color: index < root.pin.length ? "#4FC3F7" : Qt.rgba(1,1,1,0.22)
                    border.color: Qt.rgba(1,1,1,0.40); border.width: 1
                    Behavior on color { ColorAnimation { duration: 120 } }
                }
            }
        }

        Grid {
            anchors.horizontalCenter: parent.horizontalCenter
            columns: 3; spacing: 12

            Repeater {
                model: ["1","2","3","4","5","6","7","8","9","","0","⌫"]
                delegate: Rectangle {
                    width: 74; height: 50; radius: 14
                    visible: modelData !== "" || index === 9
                    color: modelData !== "" ? Qt.rgba(1,1,1,0.11) : "transparent"
                    border.color: modelData !== "" ? Qt.rgba(1,1,1,0.24) : "transparent"
                    border.width: 1

                    Rectangle {
                        visible: modelData !== ""
                        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                        height: parent.height * 0.45; radius: parent.radius
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.12) }
                            GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
                        }
                    }

                    Text {
                        anchors.centerIn: parent
                        text: modelData
                        color: "white"
                        font.pixelSize: modelData === "⌫" ? 20 : 22
                        font.weight: Font.Medium
                    }

                    scale: ma.pressed ? 0.88 : 1.0
                    Behavior on scale { NumberAnimation { duration: 70 } }

                    MouseArea {
                        id: ma
                        anchors.fill: parent
                        enabled: modelData !== ""
                        onClicked: {
                            if (modelData === "⌫") {
                                if (root.pin.length > 0)
                                    root.pin = root.pin.slice(0, -1)
                            } else if (root.pin.length < root.maxLength) {
                                root.pin += modelData
                                if (root.pin.length === root.maxLength)
                                    root.pinComplete(root.pin)
                            }
                        }
                    }
                }
            }
        }
    }
}
