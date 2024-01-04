<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $id = $_GET["id"];

    $db->query("UPDATE simulation Set NUM_DOWNLOADS = NUM_DOWNLOADS + 1, TIMESTAMP = TIMESTAMP where ID=$id");

    $db->commit();
    $db->close();
?>