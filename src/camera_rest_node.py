"""
REST-based node that interfaces with MADSci and provides a USB camera interface
"""

import tempfile
from pathlib import Path
from typing import Annotated, Optional, Union
from datetime import datetime

import cv2
from madsci.common.ownership import get_current_ownership_info
from madsci.common.types.node_types import RestNodeConfig
from madsci.common.types.resource_types import Slot
from madsci.node_module.helpers import action
from madsci.node_module.rest_node_module import RestNode
from pyzbar.pyzbar import decode


class CameraNodeConfig(RestNodeConfig):
    """Configuration for the camera node module."""

    camera_address: int = 0
    """The camera address, either a number for windows or a device path in Linux/Mac."""


class CameraNode(RestNode):
    """Node that interfaces with MADSci and provides a USB camera interface"""

    config: CameraNodeConfig = CameraNodeConfig()
    config_model = CameraNodeConfig
    camera: cv2.VideoCapture = None

    def startup_handler(self) -> None:
        """Called to (re)initialize the node. Should be used to open connections to devices or initialize any other resources."""

        # Create picture capture deck template
        capture_deck_slot = Slot(
            resource_name="camera_capture_deck",
            resource_class="CameraCaptureDeck",
            capacity=1,
            attributes={
                "slot_type": "capture_deck",
                "can_capture": True,
                "light": "on",
                "description": "Camera capture deck where items are placed for imaging",
            },
        )

        self.resource_client.init_template(
            resource=capture_deck_slot,
            template_name="camera_capture_deck_slot",
            description="Template for camera capture deck slot. Represents the position where items are placed for picture taking.",
            required_overrides=["resource_name"],
            tags=["camera", "capture", "deck", "slot", "imaging"],
            created_by=self.node_definition.node_id,
            version="1.0.0",
        )

        # Initialize capture deck resource
        deck_resource_name = "camera_capture_deck_" + str(
            self.node_definition.node_name
        )
        self.capture_deck = self.resource_client.create_resource_from_template(
            template_name="camera_capture_deck_slot",
            resource_name=deck_resource_name,
            add_to_database=True,
            overrides={"owner": get_current_ownership_info().model_dump(mode="json")},
        )
        self.logger.log(
            f"Initialized capture deck resource from template: {self.capture_deck.resource_id}"
        )

        self.camera = cv2.VideoCapture(self.config.camera_address)
        if not self.camera.isOpened():
            raise Exception("Unable to connect to camera")
        self.logger.log("Camera node initialized!")

    def state_handler(self) -> None:
        """Periodically called to update the current state of the node."""
        if self.camera is not None:
            self.node_state = {"camera_status": "connected"}
            self.logger.log("Camera is operational.")
        else:
            self.node_state = {"camera_status": "disconnected"}
            self.logger.log_warning("Camera is not connected.")

    @action
    def take_picture(
        self, focus: Optional[int] = None, autofocus: Optional[bool] = None
    ) -> Annotated[Path, "The picture taken by the camera"]:
        """Action that takes a picture using the configured camera. The focus used can be set using the focus parameter."""

        # * Handle autofocus/refocusing
        try:
            if focus is not None or autofocus is not None:
                self.logger.log_info("Adjusting focus settings")
                self.adjust_focus_settings(self.camera, focus, autofocus)
        except Exception as e:
            self.logger.log_error(f"Failed to adjust focus settings: {e}")

        success, frame = self.camera.read()
        if not success:
            # if self.camera.isOpened():
            #     self.camera.release()
            raise Exception("Unable to read from camera")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)
            cv2.imwrite(str(temp_file_path), frame)
        # self.camera.release()

        return temp_file_path

    @action
    def read_barcode(
        self,
        focus: Optional[int] = None,
        autofocus: Optional[bool] = None,
    ) -> tuple[
        Annotated[
            str, "The barcode read from the image, or None if no barcode was found"
        ],
        Annotated[Path, "The picture taken by the camera"],
    ]:
        """
        Takes an image and returns the values of any barcodes present in the image. Camera focus can be adjusted using the provided parameters if necessary.

        Args:
            camera (cv2.VideoCapture): The camera object to adjust focus for.
            focus (Optional[int]): The desired focus value (used if autofocus is disabled).
            autofocus (Optional[bool]): Whether to enable or disable autofocus.

        Returns:
            ActionSucceded regardless of if barcode is collected or not.
            Barcode field in ActionResult data dictionary will contain 'None' if no barcode was collected

        """
        try:
            # take an image and collect the image path
            image_path = self.take_picture(focus=focus, autofocus=autofocus)

            # try to collect the barcode from the image
            image = cv2.imread(image_path)
            barcode = None

            all_detected_barcodes = decode(image)
            if all_detected_barcodes:
                # Note: only collects the first in a potential list of barcodes
                barcode = all_detected_barcodes[0].data.decode("utf-8")

        except Exception as e:
            raise e

        return barcode, image_path

    @action
    def take_picture_with_timestamp(
        self, focus: Optional[int] = None, autofocus: Optional[bool] = None
    ) -> Annotated[Path, "The picture taken with a timestamp"]:
        """Take a picture and write a timestamp at the bottom-left corner.

        Returns:
            Path: Path to the newly saved timestamped image.
        """
        # Reuse existing take_picture to capture raw frame
        image_path = self.take_picture(focus=focus, autofocus=autofocus)

        image = cv2.imread(str(image_path))
        if image is None:
            raise Exception("Failed to read image for timestamping")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, image.shape[1] / 2000.0 * 0.6)
        thickness = 2

        # position near bottom-left with small margin
        text_size, _ = cv2.getTextSize(timestamp, font, scale, thickness)
        x = 10
        y = image.shape[0] - 10

        # Draw semi-opaque background rectangle for readability
        rect_tl = (x - 6, y - text_size[1] - 6)
        rect_br = (x + text_size[0] + 6, y + 6)
        cv2.rectangle(image, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)

        # Put white timestamp text
        cv2.putText(image, timestamp, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)
            cv2.imwrite(str(temp_file_path), image)

        return temp_file_path

    @action
    def capture_read_and_process(
        self, focus: Optional[int] = None, autofocus: Optional[bool] = None
    ) -> Annotated[Path, "Processed image (YAY!!! overlay or grayscaled original)"]:
        """Performs a sequence:

        1. take picture with timestamp
        2. read barcode
        3. if a qr is detected successfully:
           - take a new picture with timestamp
           - write "YAY!!!" on top of the image in bold red and return it
        4. else:
           - grayscale the originally taken timestamped picture and return that
        """
        print("><><><>>>> inside capture_read_and_process!!")
        # 1. take original timestamped picture
        original_path = self.take_picture_with_timestamp(focus=focus, autofocus=autofocus)

        # 2. read barcode (this will take a fresh picture internally in read_barcode)
        barcode, _ = self.read_barcode(focus=focus, autofocus=autofocus)
        print("><><><>>>> just read bardcode!!")
        print(f"><><><>>>> {barcode = }")

        if barcode is not None:
            # 3.1 take a new picture with timestamp
            new_path = self.take_picture_with_timestamp(focus=focus, autofocus=autofocus)
            img = cv2.imread(str(new_path))
            if img is None:
                raise Exception("Failed to read newly captured image to write YAY!!!")

            # 3.2 write "YAY!!!" on top of the image in bold red
            text = "YAY!!!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            # scale font relative to image width
            scale = max(1.0, img.shape[1] / 800.0)
            thickness = max(2, int(round(scale)))

            text_size, _ = cv2.getTextSize(text, font, scale, thickness)
            x = int((img.shape[1] - text_size[0]) / 2)
            y = 10 + text_size[1]

            # draw black outline for readability then red text
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = Path(temp_file.name)
                cv2.imwrite(str(temp_file_path), img)

        else:
            barcode = "--- QR NOT DETECTED ---"
            # 4.1 grayscale the originally taken timestamped picture and return that
            orig_img = cv2.imread(str(original_path))
            if orig_img is None:
                raise Exception("Failed to read original image to grayscale")

            gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

            # Save gray image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = Path(temp_file.name)
                cv2.imwrite(str(temp_file_path), gray)

        return barcode, temp_file_path


    def adjust_focus_settings(
        self,
        camera: cv2.VideoCapture,
        focus: Optional[int] = None,
        autofocus: Optional[bool] = None,
    ) -> None:
        """
        Adjusts the camera's focus, if necessary/possible, based on the provided parameters.

        Args:
            camera (cv2.VideoCapture): The camera object to adjust focus for.
            focus (Optional[int]): The desired focus value (used if autofocus is disabled).
            autofocus (Optional[bool]): Whether to enable or disable autofocus.

        Raises:
            Exception: If the camera does not support autofocus or manual focus.
            ValueError: If the focus value is out of range.
        """
        focus_changed = False

        if autofocus is not None:
            self.logger.log_info(f"Setting autofocus to {autofocus}")
            current_autofocus = camera.get(cv2.CAP_PROP_AUTOFOCUS)
            if current_autofocus != (1 if autofocus else 0):
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
                focus_changed = True

        if not autofocus and focus is not None:
            self.logger.log_info(f"Setting focus to {focus}")
            if focus < 0 or focus > 255:
                raise ValueError("Focus value must be between 0 and 255.")
            current_focus = camera.get(cv2.CAP_PROP_FOCUS)
            if current_focus != focus:
                camera.set(cv2.CAP_PROP_FOCUS, focus)
                focus_changed = True

        if focus_changed:
            self.logger.log_info(
                "Focus settings changed. Waiting for focus to stabilize."
            )
            for _ in range(30):  # Discard 30 frames to allow focus to stabilize
                camera.read()
        else:
            for _ in range(
                5
            ):  # Discard 5 frames in case the camera needs a moment for startup
                camera.read()


if __name__ == "__main__":
    camera_node = CameraNode()
    camera_node.start_node()
